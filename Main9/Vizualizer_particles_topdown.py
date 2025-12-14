"""Simple top-down particle visualizer.

Renders `sim.build_point_vertices()` as GL_POINTS with an orthographic
top-down camera. Can also display particle density as a 2D texture overlay.

Keys:
  Space       - pause/unpause
  N           - single-step
  R           - reseed density
  S           - print sim statistics
  P           - toggle particle density texture overlay
  G / L       - increase/decrease texture brightness
  [ / ]       - move Z slice down/up
  Up / Down   - move Z slice (alternative keys)
  Mouse Wheel - adjust Z slice (when window focused)
  T           - show particle count per Z layer
  D           - debug: print sample particle data (pos, vel, color)
  Esc         - quit

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
    gl_PointSize = 8.0;
}
"""

POINTS_FRAG = r"""
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }
"""

TEXTURE_VERT = r"""
#version 330 core
out vec2 vUV;
void main() {
    vec2 p;
    if (gl_VertexID == 0) p = vec2(-1.0, -1.0);
    else if (gl_VertexID == 1) p = vec2( 3.0, -1.0);
    else p = vec2(-1.0,  3.0);
    vUV = 0.5 * (p + 1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

TEXTURE_FRAG = r"""
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform float uGain;
void main() {
    float v = texture(uTex, vUV).r * uGain;
    v = clamp(v, 0.0, 1.0);
    FragColor = vec4(v, v, v, 1.0);
}
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


def _create_2d_texture(nx, ny):
    """Create a 2D R32F texture for particle density."""
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


def _upload_particle_density_2d(sim, tex, z_slice):
    """Render particle positions in a Z slice to a 2D texture showing density/count.
    
    z_slice: index into Z dimension [0, NZ)
    Particles within tolerance voxels of z_slice are included.
    Layers overlap to show boundary particles.
    """
    nx, ny = int(sim.NX), int(sim.NY)
    nz = int(sim.NZ)
    density = cp.zeros((ny, nx), dtype=cp.float32)

    # Extract X,Y,Z positions and normalize to [0, 1]
    half_lx = sim.LX * sim.NX / 2.0
    half_ly = sim.LY * sim.NY / 2.0
    half_lz = sim.LZ * sim.NZ / 2.0
    
    px = (sim.particles[:, 0] + half_lx) / (sim.LX * sim.NX)
    py = (sim.particles[:, 1] + half_ly) / (sim.LY * sim.NY)
    pz = (sim.particles[:, 2] + half_lz) / (sim.LZ * sim.NZ)
    
    # Clamp to valid range
    px = cp.clip(px, 0, 1.0 - 1e-6)
    py = cp.clip(py, 0, 1.0 - 1e-6)
    pz = cp.clip(pz, 0, 1.0 - 1e-6)
    
    # Map to voxel indices
    iz = (pz * nz).astype(cp.int32)
    
    # Larger tolerance for overlapping layers (shows adjacent slices too)
    z_target = int(max(0, min(nz - 1, z_slice)))
    tolerance = 2  # Show ±2 layers around current slice
    mask = cp.abs(iz - z_target) <= tolerance
    
    px_slice = px[mask]
    py_slice = py[mask]
    
    # Map to pixel indices for filtered particles
    ix = (px_slice * nx).astype(cp.int32)
    iy = (py_slice * ny).astype(cp.int32)
    
    # Accumulate density
    if len(ix) > 0:
        cp.add.at(density, (iy, ix), 1.0)
    
    # Normalize and upload
    density_np = density.astype(cp.float32, copy=False).get()
    if not density_np.flags["C_CONTIGUOUS"]:
        density_np = np.ascontiguousarray(density_np)
    
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, nx, ny, gl.GL_RED, gl.GL_FLOAT, density_np)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


def _create_dummy_vao():
    """Create a dummy VAO for fullscreen triangle rendering."""
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    gl.glBindVertexArray(0)
    return vao


def _print_z_progress(z_slice, nz):
    """Print a visual progress bar for Z slice selection."""
    bar_width = 40
    filled = int((z_slice / max(1, nz - 1)) * bar_width)
    bar = '█' * filled + '░' * (bar_width - filled)
    pct = 100 * (z_slice / max(1, nz - 1))
    print(f"\rZ: [{bar}] {z_slice:2d}/{nz-1:2d} ({pct:5.1f}%)", end='', flush=True)


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
    show_texture = False
    texture_gain = 1.0
    z_slice = 0  # current Z layer to visualize

    # Create texture and VAO for density overlay
    tex = _create_2d_texture(int(sim.NX), int(sim.NY))
    tex_vao = _create_dummy_vao()
    tex_prog = compileProgram(compileShader(TEXTURE_VERT, gl.GL_VERTEX_SHADER), compileShader(TEXTURE_FRAG, gl.GL_FRAGMENT_SHADER))

    def on_key(win, key, scancode, action, mods):
        nonlocal paused, step_once, show_texture, texture_gain, z_slice
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
        elif key == glfw.KEY_P:
            show_texture = not show_texture
        elif key == glfw.KEY_G:
            texture_gain *= 1.2
        elif key == glfw.KEY_L:
            texture_gain /= 1.2
        elif key == glfw.KEY_LEFT_BRACKET:
            z_slice = max(0, z_slice - 1)
            _print_z_progress(z_slice, int(sim.NZ))
        elif key == glfw.KEY_RIGHT_BRACKET:
            z_slice = min(int(sim.NZ) - 1, z_slice + 1)
            _print_z_progress(z_slice, int(sim.NZ))
        elif key == glfw.KEY_UP:
            z_slice = min(int(sim.NZ) - 1, z_slice + 1)
            _print_z_progress(z_slice, int(sim.NZ))
        elif key == glfw.KEY_DOWN:
            z_slice = max(0, z_slice - 1)
            _print_z_progress(z_slice, int(sim.NZ))
        elif key == glfw.KEY_T:
            # Debug: show particle count per Z layer
            nz = int(sim.NZ)
            half_lz = sim.LZ * nz / 2.0
            pz = (sim.particles[:, 2] + half_lz) / (sim.LZ * nz)
            pz = cp.clip(pz, 0, 1.0 - 1e-6)
            iz = (pz * nz).astype(cp.int32)
            print("\n\nParticle distribution by Z layer:")
            for z_idx in range(nz):
                count = int(cp.sum(iz == z_idx))
                print(f"  Z={z_idx}: {count} particles")
        elif key == glfw.KEY_D:
            # Debug: print particle 0 info
            print("\n\n=== DEBUG: Particle Data ===")
            print(f"Total particles: {len(sim.particles)}")
            print(f"Grid: NX={sim.NX}, NY={sim.NY}, NZ={sim.NZ}")
            
            # Sample first few particles
            for idx in [0, 1, min(len(sim.particles)-1, 100)]:
                pos = sim.particles[idx].get() if isinstance(sim.particles[idx], cp.ndarray) else sim.particles[idx]
                pos_prev = sim.particles_prev[idx].get() if isinstance(sim.particles_prev[idx], cp.ndarray) else sim.particles_prev[idx]
                vel = (pos - pos_prev) if isinstance(pos, np.ndarray) else (sim.particles[idx] - sim.particles_prev[idx])
                
                print(f"\nParticle {idx}:")
                print(f"  Position: {pos}")
                print(f"  Velocity: {vel if isinstance(vel, np.ndarray) else vel.get()}")
                
                # Get vertex color
                verts = sim.build_point_vertices()
                vert_data = verts[idx].get() if isinstance(verts[idx], cp.ndarray) else verts[idx]
                print(f"  Vertex (x,y,z,r,g,b): {vert_data}")
            
            print(f"\nParticles range X: [{float(cp.min(sim.particles[:, 0])):.3f}, {float(cp.max(sim.particles[:, 0])):.3f}]")
            print(f"Particles range Y: [{float(cp.min(sim.particles[:, 1])):.3f}, {float(cp.max(sim.particles[:, 1])):.3f}]")
            print(f"Particles range Z: [{float(cp.min(sim.particles[:, 2])):.3f}, {float(cp.max(sim.particles[:, 2])):.3f}]")
            
            # VBO Debug
            print(f"\n=== DEBUG: VBO Info ===")
            print(f"Points count (pts_count): {pts_count}")
            print(f"VBO stride: 6 floats = 24 bytes")
            print(f"Expected VBO size: {pts_count} * 24 = {pts_count * 24} bytes")
            
            # Check if VAO is bound correctly
            print(f"VAO ID: {vao}")
            print(f"VBO ID: {vbo}")

    glfw.set_key_callback(window, on_key)

    def on_scroll(win, xoff, yoff):
        nonlocal z_slice
        z_slice = max(0, min(int(sim.NZ) - 1, z_slice + int(yoff)))
        _print_z_progress(z_slice, int(sim.NZ))

    glfw.set_scroll_callback(window, on_scroll)

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

        # upload particle density texture if visible
        if show_texture:
            _upload_particle_density_2d(sim, tex, z_slice)

        # draw
        gl.glViewport(0, 0, *glfw.get_framebuffer_size(window))
        gl.glClearColor(0.1, 0.1, 0.12, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # render particle density texture if enabled
        if show_texture:
            gl.glUseProgram(tex_prog)
            gl.glUniform1f(gl.glGetUniformLocation(tex_prog, "uGain"), texture_gain)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glUniform1i(gl.glGetUniformLocation(tex_prog, "uTex"), 0)
            gl.glBindVertexArray(tex_vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            gl.glBindVertexArray(0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

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
