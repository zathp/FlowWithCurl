import cupy as cp
import numpy as np


def _init_density(sim):
    d = getattr(sim, "densityfield", None)
    if d is None or not isinstance(d, cp.ndarray) or d.ndim != 3:
        raise RuntimeError("sim.densityfield must be a CuPy array shaped (NZ,NY,NX)")

    nz, ny, nx = d.shape
    z, y, x = cp.meshgrid(
        cp.linspace(-1, 1, nz),
        cp.linspace(-1, 1, ny),
        cp.linspace(-1, 1, nx),
        indexing="ij",
    )
    blob = cp.exp(-(x * x + y * y + z * z) * 8.0)
    noise = 0.10 * cp.random.random((nz, ny, nx), dtype=cp.float32)
    d[:] = (blob + noise).astype(cp.float32)


def _make_topdown_image(sim, field: str, z_index: int, mode: str):
    if field == "density":
        vol = sim.densityfield
    elif field == "flowmag":
        v = sim.flowfield
        vol = cp.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2 + v[..., 2] ** 2)
    elif field == "curlmag":
        c = sim.curlfield
        vol = cp.sqrt(c[..., 0] ** 2 + c[..., 1] ** 2 + c[..., 2] ** 2)
    else:
        vol = sim.densityfield

    nz = int(vol.shape[0])
    zi = int(max(0, min(nz - 1, z_index)))

    if mode == "max":
        img = cp.max(vol, axis=0)
    else:
        img = vol[zi, :, :]

    return img.astype(cp.float32, copy=False).get()
