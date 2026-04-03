import argparse
import json
import pickle
import socket
import struct
import zlib

import matplotlib.pyplot as plt
import numpy as np


def parse_region_values(values):
    if values is None:
        return None

    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in values]
    return {
        "x": (min(xmin, xmax), max(xmin, xmax)),
        "y": (min(ymin, ymax), max(ymin, ymax)),
        "z": (min(zmin, zmax), max(zmin, zmax)),
    }


def recvall(sock, num_bytes):
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_payload(sock):
    header = recvall(sock, 4)
    if not header:
        return None

    size = struct.unpack("!I", header)[0]
    body = recvall(sock, size)
    if not body:
        return None

    return pickle.loads(zlib.decompress(body))


def field_to_image(field_name, data, transfer_mode="cube"):
    arr = np.asarray(data)
    if arr.size == 0:
        return None

    if field_name in ("flow", "curl") and arr.ndim == 4:
        arr = np.linalg.norm(arr, axis=-1)

    arr = np.squeeze(arr)
    if transfer_mode == "slice":
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        return np.atleast_2d(arr)

    if arr.ndim == 3:
        return arr[arr.shape[0] // 2]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return np.atleast_2d(arr)


def main():
    parser = argparse.ArgumentParser(description="Monitor client for Main11Server headless field streaming")
    parser.add_argument("--host", required=True, help="Server IP or hostname")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--field", choices=["density", "density2", "flow", "curl"], default="density")
    parser.add_argument("--transfer-mode", choices=["cube", "slice"], default="cube")
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--send-every", type=int, default=20)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--region", nargs=6, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument("--no-plot", action="store_true", help="Print frame stats only")
    args = parser.parse_args()

    subscription = {
        "field": args.field,
        "transfer_mode": args.transfer_mode,
        "slice_axis": args.slice_axis,
        "slice_index": args.slice_index,
        "region": parse_region_values(args.region),
        "stride": max(1, args.stride),
        "send_every": max(1, args.send_every),
        "max_points": max(1, args.max_points),
    }

    sock = socket.create_connection((args.host, args.port), timeout=10)
    sock.sendall((json.dumps(subscription) + "\n").encode("utf-8"))
    sock.settimeout(None)

    print(f"Connected to {args.host}:{args.port}")
    print(f"Requested field: {subscription['field']}")
    print(f"Requested transfer mode: {subscription['transfer_mode']}")
    print(f"Requested region: {subscription['region']}")

    if args.no_plot:
        while True:
            payload = recv_payload(sock)
            if payload is None:
                print("Server disconnected")
                break
            print(
                f"frame={payload['frame_id']:6d} step={payload['step_count']:6d} "
                f"mode={payload.get('transfer_mode', 'cube'):5s} field={payload['field_name']:8s} "
                f"cells={payload['num_cells']:6d} shape={payload['shape']}"
            )
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(np.zeros((2, 2), dtype=np.float32), cmap="viridis", origin="lower")
    cbar = plt.colorbar(image, ax=ax)
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.set_xlabel("X index")
    ax.set_ylabel("Y index")

    while True:
        payload = recv_payload(sock)
        if payload is None:
            print("Server disconnected")
            break

        frame = field_to_image(payload["field_name"], payload["field_data"], payload.get("transfer_mode", "cube"))
        if frame is None or frame.size == 0:
            ax.set_title(f"Frame {payload['frame_id']} - no cells in requested region")
            plt.pause(0.001)
            continue

        image.set_data(frame)
        image.set_clim(vmin=float(np.min(frame)), vmax=float(np.max(frame) + 1e-9))
        ax.set_title(
            f"Frame {payload['frame_id']} | step {payload['step_count']} | "
            f"{payload.get('transfer_mode', 'cube')} {payload['field_name']} | cells {payload['num_cells']}"
        )
        ax.set_aspect("auto")
        plt.pause(0.001)

    sock.close()


if __name__ == "__main__":
    main()
