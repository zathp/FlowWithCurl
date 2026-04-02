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


def main():
    parser = argparse.ArgumentParser(description="Monitor client for Main11Server headless particle streaming")
    parser.add_argument("--host", required=True, help="Server IP or hostname")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--send-every", type=int, default=20)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--region", nargs=6, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument("--no-plot", action="store_true", help="Print frame stats only")
    args = parser.parse_args()

    subscription = {
        "region": parse_region_values(args.region),
        "stride": max(1, args.stride),
        "send_every": max(1, args.send_every),
        "max_points": max(1, args.max_points),
    }

    sock = socket.create_connection((args.host, args.port), timeout=10)
    sock.sendall((json.dumps(subscription) + "\n").encode("utf-8"))
    sock.settimeout(None)

    print(f"Connected to {args.host}:{args.port}")
    print(f"Requested region: {subscription['region']}")

    if args.no_plot:
        while True:
            payload = recv_payload(sock)
            if payload is None:
                print("Server disconnected")
                break
            print(
                f"frame={payload['frame_id']:6d} step={payload['step_count']:6d} "
                f"verts={payload['num_vertices']:6d} density_mean={payload['stats']['density']['mean']:.4f}"
            )
        return

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter([], [], s=[], c=[])
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    while True:
        payload = recv_payload(sock)
        if payload is None:
            print("Server disconnected")
            break

        verts = payload["verts"]
        if verts.shape[0] == 0:
            ax.set_title(f"Frame {payload['frame_id']} - no vertices in requested region")
            plt.pause(0.001)
            continue

        pos = verts[:, :2]
        colors = verts[:, 3:6]
        sizes = np.clip(verts[:, 6], 1.0, 50.0)

        scatter.remove()
        scatter = ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=sizes, edgecolors="none")
        ax.set_title(f"Frame {payload['frame_id']}  |  step {payload['step_count']}  |  verts {payload['num_vertices']}")
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

    sock.close()


if __name__ == "__main__":
    main()
