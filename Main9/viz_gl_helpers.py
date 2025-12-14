import ctypes
import numpy as np
import cupy as cp
import OpenGL.GL as gl


def _create_fullscreen_vao():
    verts = np.array([
        -1.0, -1.0, 0.0,
         3.0, -1.0, 0.0,
        -1.0,  3.0, 0.0,
    ], dtype=np.float32)

    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)

    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 0, None)
    gl.glBindVertexArray(0)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
    return vao


def _create_density_3d_texture(nx, ny, nz):
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_3D, tex)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, gl.GL_R32F, nx, ny, nz, 0, gl.GL_RED, gl.GL_FLOAT, None)

    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)

    gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
    return tex


def _upload_density_3d(tex, density_cp):
    density_np = density_cp.astype(cp.float32, copy=False).get()
    if not density_np.flags["C_CONTIGUOUS"]:
        density_np = np.ascontiguousarray(density_np)

    nz, ny, nx = density_np.shape
    gl.glBindTexture(gl.GL_TEXTURE_3D, tex)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexSubImage3D(gl.GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz, gl.GL_RED, gl.GL_FLOAT, density_np)
    gl.glBindTexture(gl.GL_TEXTURE_3D, 0)


def _create_scalar_2d_texture(nx, ny):
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, nx, ny, 0, gl.GL_RED, gl.GL_FLOAT, None)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tex


def _upload_scalar_2d(tex, img_np):
    if img_np.dtype != np.float32:
        img_np = img_np.astype(np.float32, copy=False)
    if not img_np.flags["C_CONTIGUOUS"]:
        img_np = np.ascontiguousarray(img_np)

    h, w = img_np.shape
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h, gl.GL_RED, gl.GL_FLOAT, img_np)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


def _setup_points(sim):
    nx, ny, nz = int(sim.NX), int(sim.NY), int(sim.NZ)
    n = nx * ny * nz

    float_size = np.float32().nbytes
    stride = 6 * float_size

    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, n * stride, None, gl.GL_DYNAMIC_DRAW)

    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)

    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))

    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * float_size))

    gl.glBindVertexArray(0)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    return vao, vbo, n


def _upload_points(sim, vbo):
    pts = sim.build_point_vertices()  # expected (N,6) CuPy
    if not isinstance(pts, cp.ndarray) or pts.ndim != 2 or pts.shape[1] != 6:
        raise RuntimeError("build_point_vertices() must return CuPy array shaped (N,6)")

    pts_np = pts.astype(cp.float32, copy=False).get()
    if not pts_np.flags["C_CONTIGUOUS"]:
        pts_np = np.ascontiguousarray(pts_np)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, pts_np.nbytes, pts_np)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
