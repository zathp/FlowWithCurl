import argparse
import json
import pickle
import socket
import struct
import time
import traceback
import zlib

import cupy as cp

from WorldStep import WorldStep


def compute_nz_partition(total_nz, partition_index=0, partition_count=1):
    partition_count = max(1, int(partition_count))
    partition_index = max(0, min(int(partition_index), partition_count - 1))
    base = total_nz // partition_count
    remainder = total_nz % partition_count
    start = partition_index * base + min(partition_index, remainder)
    size = base + (1 if partition_index < remainder else 0)
    return start, size


def create_sim(device_id=0, nz_partitions=1, partition_index=0, halo_dir=None, halo_timeout=10.0):
    global_nz = 100
    z_start, local_nz = compute_nz_partition(global_nz, partition_index=partition_index, partition_count=nz_partitions)
    return WorldStep(
        nx=100,
        ny=100,
        nz=local_nz,
        lx=0.1,
        ly=0.1,
        lz=0.1,
        seed=22,
        dispersion=0.1,
        damping=0.00,
        k1_size=3,
        k2_size=2,
        k3_size=2,
        k4_size=2,
        k5_size=2,
        enable_particles=False,
        density1_injection_strength_neg=1,
        density1_injection_strength_pos=1,
        device_id=device_id,
        global_nz_start=z_start,
        global_nz_total=global_nz,
        partition_index=partition_index,
        partition_count=nz_partitions,
        halo_exchange_dir=halo_dir,
        halo_timeout_s=halo_timeout,
    )


def parse_region_values(values):
    if values is None:
        return None

    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in values]
    return {
        "x": (min(xmin, xmax), max(xmin, xmax)),
        "y": (min(ymin, ymax), max(ymin, ymax)),
        "z": (min(zmin, zmax), max(zmin, zmax)),
    }


def normalize_region_dict(region):
    if not region:
        return None

    normalized = {}
    for axis in ("x", "y", "z"):
        bounds = region.get(axis)
        if bounds is None:
            continue
        lo, hi = float(bounds[0]), float(bounds[1])
        normalized[axis] = (min(lo, hi), max(lo, hi))
    return normalized or None


def recv_client_config(conn, timeout=2.0):
    conn.settimeout(timeout)
    try:
        file_obj = conn.makefile("rb")
        line = file_obj.readline(65536)
        if not line:
            return {}
        return json.loads(line.decode("utf-8").strip())
    except Exception:
        return {}
    finally:
        conn.settimeout(None)


def send_payload(conn, payload):
    packed = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(packed, level=1)
    conn.sendall(struct.pack("!I", len(compressed)))
    conn.sendall(compressed)


def build_stream_payload(sim, frame_id, field_name="density", region=None, stride=1, max_points=25000,
                         transfer_mode="cube", slice_axis="z", slice_index=None):
    block = sim.extract_field_block(
        field_name=field_name,
        region=region,
        stride=stride,
        max_cells=max_points,
        transfer_mode=transfer_mode,
        slice_axis=slice_axis,
        slice_index=slice_index,
    )
    data_np = cp.asnumpy(block["data"]).astype("float32", copy=False)
    spatial_shape = tuple(int(v) for v in block["spatial_shape"])
    num_cells = int(spatial_shape[0] * spatial_shape[1] * spatial_shape[2])

    return {
        "type": "field_frame",
        "field_name": field_name,
        "frame_id": int(frame_id),
        "step_count": int(sim.step_count),
        "num_cells": num_cells,
        "shape": tuple(int(v) for v in data_np.shape),
        "region": region,
        "stride": int(block["stride"]),
        "transfer_mode": block["transfer_mode"],
        "slice_axis": block["slice_axis"],
        "slice_index": block["slice_index"],
        "index_bounds": block["index_bounds"],
        "world_bounds": block["world_bounds"],
        "field_data": data_np,
        "stats": {
            "density": sim.get_field_stats("density"),
            "flow": sim.get_field_stats("flow"),
            "curl": sim.get_field_stats("curl"),
        },
    }


def run_headless_server(host="0.0.0.0", port=5055, send_every=20, stride=1, region=None, max_points=25000,
                        sim_ups=240, sim_dt=0.05, device_id=0, max_steps=0,
                        field_name="density", nz_partitions=1, partition_index=0,
                        halo_dir=None, halo_timeout=10.0,
                        transfer_mode="cube", slice_axis="z", slice_index=None):
    gpu_count = cp.cuda.runtime.getDeviceCount()
    print(f"CuPy detected {gpu_count} CUDA GPU(s); using device {device_id}")

    sim = create_sim(
        device_id=device_id,
        nz_partitions=nz_partitions,
        partition_index=partition_index,
        halo_dir=halo_dir,
        halo_timeout=halo_timeout,
    )
    print(
        f"Headless field simulation initialized successfully "
        f"(local NZ={sim.NZ}, global_z_start={sim.global_nz_start}, global_NZ={sim.global_nz_total})"
    )
    if nz_partitions > 1:
        print(f"Halo exchange enabled for partition {partition_index}/{nz_partitions - 1} via {sim.halo_exchange_dir}")
    print(f"Grid cells available for streaming: {sim.NX * sim.NY * sim.NZ}")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen()
    server_sock.settimeout(0.01)

    print(f"Streaming server listening on {host}:{port}")
    print("Clients can optionally send one JSON line on connect, for example:")
    print('{"field":"density","transfer_mode":"slice","slice_axis":"z","slice_index":55,"region":{"x":[-2,2],"y":[-2,2],"z":[-2,2]},"stride":2,"send_every":10,"max_points":12000}')

    clients = []
    frame_id = 0
    next_step_time = time.perf_counter()
    step_interval = 1.0 / max(1, int(sim_ups))

    try:
        while True:
            try:
                conn, addr = server_sock.accept()
                config = recv_client_config(conn)
                client_info = {
                    "sock": conn,
                    "addr": addr,
                    "field_name": str(config.get("field", field_name)),
                    "transfer_mode": str(config.get("transfer_mode", transfer_mode)),
                    "slice_axis": str(config.get("slice_axis", slice_axis)),
                    "slice_index": config.get("slice_index", slice_index),
                    "region": normalize_region_dict(config.get("region")) or region,
                    "stride": max(1, int(config.get("stride", stride))),
                    "send_every": max(1, int(config.get("send_every", send_every))),
                    "max_points": max(1, int(config.get("max_points", max_points))),
                }
                clients.append(client_info)
                print(
                    f"Client connected from {addr} with field={client_info['field_name']} "
                    f"mode={client_info['transfer_mode']} region={client_info['region']}"
                )
            except socket.timeout:
                pass

            now = time.perf_counter()
            if now < next_step_time:
                time.sleep(min(0.002, next_step_time - now))
                continue

            sim.step(sim_dt, print_timings=False)
            next_step_time += step_interval

            dead_clients = []
            for client in clients:
                if sim.step_count % client["send_every"] != 0:
                    continue

                try:
                    payload = build_stream_payload(
                        sim,
                        frame_id=frame_id,
                        field_name=client["field_name"],
                        region=client["region"],
                        stride=client["stride"],
                        max_points=client["max_points"],
                        transfer_mode=client["transfer_mode"],
                        slice_axis=client["slice_axis"],
                        slice_index=client["slice_index"],
                    )
                    send_payload(client["sock"], payload)
                except Exception as send_error:
                    print(f"Client disconnected {client['addr']}: {send_error}")
                    dead_clients.append(client)

            for client in dead_clients:
                try:
                    client["sock"].close()
                except Exception:
                    pass
                if client in clients:
                    clients.remove(client)

            if sim.step_count % max(1, send_every * 5) == 0:
                print(f"step={sim.step_count:6d} clients={len(clients)} frame={frame_id}")

            frame_id += 1
            if max_steps > 0 and sim.step_count >= max_steps:
                print(f"Reached max_steps={max_steps}; stopping server loop")
                break
    finally:
        for client in clients:
            try:
                client["sock"].close()
            except Exception:
                pass
        server_sock.close()


def run_local_viewer(*args, **kwargs):
    raise RuntimeError("Viewer mode is disabled in Main11Server; use the headless server plus monitor_client.py")


def main():
    parser = argparse.ArgumentParser(description="Main11Server headless field streaming server")
    parser.add_argument("--mode", choices=["server", "viewer"], default="server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--field", choices=["density", "density2", "flow", "curl"], default="density")
    parser.add_argument("--transfer-mode", choices=["cube", "slice"], default="cube", help="Send either a 3D cube subset or a single 2D slice")
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default="z", help="Axis used when --transfer-mode slice is selected")
    parser.add_argument("--slice-index", type=int, default=None, help="Grid index for slice mode; for z this is the global z index")
    parser.add_argument("--send-every", type=int, default=20, help="Send one frame every N simulation steps")
    parser.add_argument("--stride", type=int, default=1, help="Subsample streamed grid cells by this stride")
    parser.add_argument("--max-points", type=int, default=25000, help="Cap streamed grid cells per frame")
    parser.add_argument("--region", nargs=6, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"), help="World-space bounds for the small cube or slice window")
    parser.add_argument("--sim-ups", type=int, default=240)
    parser.add_argument("--sim-dt", type=float, default=0.05)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--nz-partitions", type=int, default=1, help="Divide the total Z depth into this many slabs")
    parser.add_argument("--partition-index", type=int, default=0, help="Which NZ slab this process owns")
    parser.add_argument("--halo-dir", default=None, help="Shared directory used for halo exchange between NZ partitions")
    parser.add_argument("--halo-timeout", type=float, default=10.0, help="Seconds to wait for neighbor halo files")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop automatically after N steps (0 = run forever)")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()

    try:
        region = parse_region_values(args.region)
        if args.mode == "viewer":
            run_local_viewer()
        else:
            run_headless_server(
                host=args.host,
                port=args.port,
                send_every=args.send_every,
                stride=args.stride,
                region=region,
                max_points=args.max_points,
                sim_ups=args.sim_ups,
                sim_dt=args.sim_dt,
                device_id=args.device_id,
                max_steps=args.max_steps,
                field_name=args.field,
                nz_partitions=args.nz_partitions,
                partition_index=args.partition_index,
                halo_dir=args.halo_dir,
                halo_timeout=args.halo_timeout,
                transfer_mode=args.transfer_mode,
                slice_axis=args.slice_axis,
                slice_index=args.slice_index,
            )
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
