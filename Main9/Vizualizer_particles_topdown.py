"""Simple top-down particle visualizer.

Renders `sim.build_point_vertices()` as GL_POINTS with an orthographic
top-down camera. Keys:
  Space - pause/unpause
  N     - single-step
  R     - reseed density (calls `sim.init_densityfield()` if present)
  S     - print sim statistics (`sim.print_field_stats()`)
  Esc   - quit

This file is intentionally compact and reuses small helpers from the
existing visualizer code (VBO uploads / simple shaders).
"""
import ctypes
import numpy as np
import glfw
import cupy as cp
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from WorldStep import WorldStep


POINTS_VERT = r"""
#version 330 core
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inColor;
uniform mat4 uMVP;
out vec3 vColor;
void main() {
    gl_Position = uMVP * vec4(inPos, 1.0);
    vColor = inColor;
    gl_PointSize = 3.0;
}
"""

POINTS_FRAG = r"""
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }
"""


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
    pts = sim.build_point_vertices()
    if not isinstance(pts, cp.ndarray):
        raise RuntimeError("build_point_vertices() must return a CuPy array")
    pts_np = pts.astype(cp.float32, copy=False).get()
    if not pts_np.flags["C_CONTIGUOUS"]:
        pts_np = np.ascontiguousarray(pts_np)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, pts_np.nbytes, pts_np)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)


def run_viewer(sim: WorldStep):
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW")

    window = glfw.create_window(900, 900, "Top-down Particles", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    prog = compileProgram(compileShader(POINTS_VERT, gl.GL_VERTEX_SHADER), compileShader(POINTS_FRAG, gl.GL_FRAGMENT_SHADER))

    vao, vbo, pts_count = _setup_points(sim)

    paused = True
    step_once = False

    def on_key(win, key, scancode, action, mods):
        nonlocal paused, step_once
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            paused = not paused
        elif key == glfw.KEY_N:
            step_once = True
        elif key == glfw.KEY_R:
            if hasattr(sim, 'init_densityfield'):
                sim.init_densityfield()
        elif key == glfw.KEY_S:
            if hasattr(sim, 'print_field_stats'):
                sim.print_field_stats()

    glfw.set_key_callback(window, on_key)

    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    # Compute orthographic projection that frames the sim in X/Y
    width = float(sim.LX * sim.NX)
    height = float(sim.LY * sim.NY)
    left = -0.5 * width
    right = 0.5 * width
    bottom = -0.5 * height
    top = 0.5 * height
    proj = pyrr.matrix44.create_orthogonal_projection(left, right, bottom, top, -1000.0, 1000.0, dtype=np.float32)

    # camera above looking down -Z
    eye = np.array([0.0, 0.0, float(sim.LZ * sim.NZ)], dtype=np.float32)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view = pyrr.matrix44.create_look_at(eye, center, up, dtype=np.float32)
    mvp = proj @ view

    while not glfw.window_should_close(window):
        glfw.poll_events()

        if (not paused) or step_once:
            if hasattr(sim, 'step'):
                sim.step(0.05)
            step_once = False

        # upload point data
        _upload_points(sim, vbo)

        # draw
        gl.glViewport(0, 0, *glfw.get_framebuffer_size(window))
        gl.glClearColor(0.02, 0.02, 0.02, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glUseProgram(prog)
        loc = gl.glGetUniformLocation(prog, "uMVP")
        gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, mvp.astype(np.float32, copy=False))

        gl.glBindVertexArray(vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, pts_count)
        gl.glBindVertexArray(0)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == '__main__':
    sim = WorldStep(nx=100, ny=100, nz=1, lx=1.0, ly=1.0, lz=1.0, seed=0, dispersion=0.1, particle_mass=20.0, k1_size=5)
    sim.print_field_stats()
    run_viewer(sim)
