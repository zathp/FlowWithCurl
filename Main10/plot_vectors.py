"""Plot vector fields (flow or curl) from WorldStep using matplotlib 3D quiver.

Usage:
    python plot_vectors.py [--field flow|curl] [--nx NX --ny NY --nz NZ] [--stride S]
    python plot_vectors.py --load-dump [--field flow|curl] [--stride S]

By default this creates a small WorldStep instance, seeds the flow, and draws a sampled quiver.
Use --load-dump to load from the temp dump created by pressing 'D' in viz_points_3d.py.
If you want to plot the live simulation state, import `plot_field(sim, ...)` and call it with your `sim`.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tempfile
import os

try:
    import cupy as cp
except Exception:
    cp = None

from WorldStep import WorldStep


def plot_field(sim, field_name="flow", stride=1, scale=1.0, cmap=None):
    """Plot a sampled 3D quiver of `field_name` from `sim`.

    sim: WorldStep instance
    field_name: 'flow' or 'curl'
    stride: integer subsampling step for each axis
    scale: arrow scale multiplier
    """
    if field_name == "flow":
        field_cp = sim.flowfield
    else:
        field_cp = sim.curlfield

    # copy to host if using CuPy
    if cp is not None and isinstance(field_cp, cp.ndarray):
        field = cp.asnumpy(field_cp)
    else:
        field = np.asarray(field_cp)

    nz, ny, nx, _ = field.shape

    ox = -sim.LX * (sim.NX - 1) / 2.0
    oy = -sim.LY * (sim.NY - 1) / 2.0
    oz = -sim.LZ * (sim.NZ - 1) / 2.0

    xs, ys, zs, us, vs, ws = [], [], [], [], [], []

    for iz in range(0, nz, stride):
        for iy in range(0, ny, stride):
            for ix in range(0, nx, stride):
                x = ox + ix * sim.LX
                y = oy + iy * sim.LY
                z = oz + iz * sim.LZ
                vec = field[iz, iy, ix]
                xs.append(x)
                ys.append(y)
                zs.append(z)
                us.append(vec[0])
                vs.append(vec[1])
                ws.append(vec[2])

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    zs = np.array(zs, dtype=np.float32)
    us = np.array(us, dtype=np.float32)
    vs = np.array(vs, dtype=np.float32)
    ws = np.array(ws, dtype=np.float32)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # normalize vectors for plotting clarity
    mags = np.sqrt(us**2 + vs**2 + ws**2) + 1e-9
    u_n = us / mags
    v_n = vs / mags
    w_n = ws / mags

    # scale arrows by magnitude
    arrow_len = scale * np.maximum(sim.LX, np.maximum(sim.LY, sim.LZ)) * 0.5
    # color by magnitude if desired
    c = mags if cmap is None else None

    ax.quiver(xs, ys, zs, u_n, v_n, w_n, length=arrow_len, normalize=False, linewidths=0.6, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Vector field: {field_name} (stride={stride})")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--field', choices=('flow', 'curl'), default='flow')
    p.add_argument('--load-dump', action='store_true', help='Load from temp dump file instead of creating new sim')
    p.add_argument('--nx', type=int, default=24)
    p.add_argument('--ny', type=int, default=24)
    p.add_argument('--nz', type=int, default=8)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--scale', type=float, default=1.0)
    args = p.parse_args()

    if args.load_dump:
        # Load from temp dump file
        dump_path = os.path.join(tempfile.gettempdir(), 'flowstep_dump.npz')
        if not os.path.exists(dump_path):
            print(f"Dump file not found: {dump_path}")
            print("Press 'D' in the visualizer to create one.")
            exit(1)
        
        data = np.load(dump_path)
        # reconstruct a minimal sim-like object
        class SimProxy:
            pass
        sim = SimProxy()
        sim.flowfield = data['flowfield']
        sim.curlfield = data['curlfield']
        sim.densityfield = data['densityfield']
        sim.NX = int(data['NX'])
        sim.NY = int(data['NY'])
        sim.NZ = int(data['NZ'])
        sim.LX = float(data['LX'])
        sim.LY = float(data['LY'])
        sim.LZ = float(data['LZ'])
        print(f"Loaded from {dump_path}")
    else:
        sim = WorldStep(nx=args.nx, ny=args.ny, nz=args.nz, lx=1.0, ly=1.0, lz=1.0)
        # flow is initialized in constructor; curl field will be computed on step
        if args.field == 'curl':
            sim.step_curlfield(dt=0.0)

    plot_field(sim, field_name=args.field, stride=args.stride, scale=args.scale)
