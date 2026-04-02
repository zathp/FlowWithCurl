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


def create_sim(device_id=0):
    return WorldStep(
        nx=100,
        ny=100,
        nz=100,
        lx=0.1,
        ly=0.1,
        lz=0.1,
        seed=22,
        dispersion=0.1,
        damping=0.00,
        particle_mass1=1000,
        particle_mass2=0.1,
        particle_dispersion=5,
        k1_size=3,
        k2_size=2,
        k3_size=2,
        k4_size=2,
        k5_size=2,
        enable_particles=True,
        density1_injection_strength_neg=1,
        density1_injection_strength_pos=1,
        device_id=device_id,
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


def build_stream_payload(sim, frame_id, region=None, stride=1, max_points=25000):
    verts_cp = sim.build_point_vertices_region(region=region, stride=stride)
    if max_points and verts_cp.shape[0] > max_points:
        sample_stride = max(1, int(verts_cp.shape[0] / max_points))
        verts_cp = verts_cp[::sample_stride]

    verts_np = cp.asnumpy(verts_cp).astype("float32", copy=False)
    return {
        "type": "frame",
        "frame_id": int(frame_id),
        "step_count": int(sim.step_count),
        "num_vertices": int(verts_np.shape[0]),
        "shape": tuple(int(v) for v in verts_np.shape),
        "region": region,
        "verts": verts_np,
        "stats": {
            "density": sim.get_field_stats("density"),
            "flow": sim.get_field_stats("flow"),
            "curl": sim.get_field_stats("curl"),
        },
    }


def run_headless_server(host="0.0.0.0", port=5055, send_every=20, stride=1, region=None, max_points=25000,
                        sim_ups=240, sim_dt=0.05, device_id=0, max_steps=0):
    gpu_count = cp.cuda.runtime.getDeviceCount()
    print(f"CuPy detected {gpu_count} CUDA GPU(s); using device {device_id}")

    sim = create_sim(device_id=device_id)
    print("Headless simulation initialized successfully")
    print(f"Particles available for streaming: {sim.num_particles * 2}")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen()
    server_sock.settimeout(0.01)

    print(f"Streaming server listening on {host}:{port}")
    print("Clients can optionally send one JSON line on connect, for example:")
    print('{"region":{"x":[-2,2],"y":[-2,2],"z":[-2,2]},"stride":2,"send_every":10,"max_points":12000}')

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
                    "region": normalize_region_dict(config.get("region")) or region,
                    "stride": max(1, int(config.get("stride", stride))),
                    "send_every": max(1, int(config.get("send_every", send_every))),
                    "max_points": max(1, int(config.get("max_points", max_points))),
                }
                clients.append(client_info)
                print(f"Client connected from {addr} with region={client_info['region']}")
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
                        region=client["region"],
                        stride=client["stride"],
                        max_points=client["max_points"],
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


def run_local_viewer(sim_ups=1000, device_id=0, snapshot=False):
    from viz_points_3d import run_viewer

    sim = create_sim(device_id=device_id)
    print("Simulation initialized successfully")
    sim.enable_particle_tracking(num_particles_to_track=10)

    if snapshot:
        from viz_2d_snapshot import save_2d_snapshot
        save_2d_snapshot(sim, "my_particles.png", dpi=200)

    print("Starting viewer...")
    print("Press 'C' in viewer to export particle force diagnostics CSV")
    run_viewer(sim, width=1920, height=1080, sim_ups=sim_ups)


def main():
    parser = argparse.ArgumentParser(description="Main11Server headless particle streaming server")
    parser.add_argument("--mode", choices=["server", "viewer"], default="server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--send-every", type=int, default=20, help="Send one frame every N simulation steps")
    parser.add_argument("--stride", type=int, default=1, help="Subsample streamed vertices by this stride")
    parser.add_argument("--max-points", type=int, default=25000, help="Cap streamed vertices per frame")
    parser.add_argument("--region", nargs=6, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument("--sim-ups", type=int, default=240)
    parser.add_argument("--sim-dt", type=float, default=0.05)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop automatically after N steps (0 = run forever)")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()

    try:
        region = parse_region_values(args.region)
        if args.mode == "viewer":
            run_local_viewer(sim_ups=args.sim_ups, device_id=args.device_id, snapshot=args.snapshot)
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
            )
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
