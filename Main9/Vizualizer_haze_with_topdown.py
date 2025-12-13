"""
Vizualizer_haze_with_topdown.py

Wrapper visualizer: two windows (shared OpenGL context)

Main window:
- Skybox (debug colors)
- Volumetric haze from sim.densityfield (NZ,NY,NX)
- Optional point cloud overlay from sim.build_point_vertices() -> (N,6) [x,y,z,r,g,b]
- Orbit camera

Top-down window:
- Orthographic 2D view (NY,NX) derived from:
  1: density
  2: |flow| magnitude (from sim.flowfield (NZ,NY,NX,3))
  3: |curl| magnitude (from sim.curlfield (NZ,NY,NX,3))
- M toggles slice/max projection
- [ ] adjusts Z slice
- +/- gain

Controls (either window focused):
- Space : pause / unpause (starts paused)
- N     : single-step
- R     : reseed density (so haze is never black due to uninitialized cp.empty)
- B     : toggle skybox
- P     : toggle point overlay
- 1/2/3 : choose top-down field
- M     : slice <-> max
- [ / ] : z slice down/up
- - / = : gain down/up
- Esc   : quit (closes both)

No wildcard imports.
"""

import math
import ctypes
import numpy as np

import glfw
import pyrr
import cupy as cp

import OpenGL.GL as gl
from OpenGL.GL.shaders import compileProgram, compileShader


# ============================================================
# Shaders
# ============================================================

SKYBOX_VERT = r"""
#version 330 core
layout (location = 0) in vec3 aPos;
out vec3 vDir;
uniform mat4 uInvViewProj;
void main() {
    vec4 p = vec4(aPos, 1.0);
    vec4 w = uInvViewProj * p;
    vDir = w.xyz / w.w;
    gl_Position = p;
}
"""

SKYBOX_FRAG = r"""
#version 330 core
in vec3 vDir;
out vec4 FragColor;
void main() {
    vec3 d = normalize(vDir);
    vec3 c;
    if (abs(d.x) > abs(d.y) && abs(d.x) > abs(d.z))
        c = (d.x > 0.0) ? vec3(1,0,0) : vec3(0,1,0);   // +X red / -X green
    else if (abs(d.y) > abs(d.z))
        c = (d.y > 0.0) ? vec3(0,0,1) : vec3(1,1,0);   // +Y blue / -Y yellow
    else
        c = (d.z > 0.0) ? vec3(1,0,1) : vec3(0,1,1);   // +Z magenta / -Z cyan
    FragColor = vec4(c, 1.0);
}
"""

FS_TRI_VERT = r"""
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

VOLUME_FRAG = r"""
#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler3D uDensityTex;
uniform mat4 uInvViewProj;
uniform vec3 uCamPos;

uniform vec3 uBoxMin;
uniform vec3 uBoxMax;

uniform float uAbsorption;
uniform int   uSteps;
uniform float uDensityScale;
uniform float uDensityBias;

bool intersectAABB(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax, out vec2 tHit) {
    vec3 invD = 1.0 / rd;
    vec3 t0 = (bmin - ro) * invD;
    vec3 t1 = (bmax - ro) * invD;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tN = max(max(tmin.x, tmin.y), tmin.z);
    float tF = min(min(tmax.x, tmax.y), tmax.z);
    tHit = vec2(tN, tF);
    return (tF >= max(tN, 0.0));
}

void main() {
    vec2 ndc = vUV * 2.0 - 1.0;
    vec4 farP = uInvViewProj * vec4(ndc, 1.0, 1.0);
    vec3 pFar = farP.xyz / farP.w;

    vec3 ro = uCamPos;
    vec3 rd = normalize(pFar - ro);

    vec2 tHit;
    if (!intersectAABB(ro, rd, uBoxMin, uBoxMax, tHit)) {
        FragColor = vec4(0.0);
        return;
    }

    float t0 = max(tHit.x, 0.0);
    float t1 = tHit.y;
    float dt = (t1 - t0) / float(max(uSteps, 1));

    float alpha = 0.0;
    vec3 col = vec3(0.0);

    for (int i = 0; i < uSteps; ++i) {
        float t = t0 + (float(i) + 0.5) * dt;
        vec3 p = ro + rd * t;

        vec3 uvw = (p - uBoxMin) / (uBoxMax - uBoxMin);
        float d = texture(uDensityTex, uvw).r;
        d = max(0.0, d * uDensityScale + uDensityBias);

        float a = 1.0 - exp(-uAbsorption * d * dt);
        vec3 sampleCol = vec3(1.0);

        float oneMinus = (1.0 - alpha);
        col += sampleCol * a * oneMinus;
        alpha += a * oneMinus;

        if (alpha > 0.995) break;
    }

    FragColor = vec4(col, alpha);
}
"""

POINTS_VERT = r"""
#version 330 core
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inColor;
uniform mat4 uMVP;
out vec3 vColor;
void main() {
    gl_Position = uMVP * vec4(inPos, 1.0);
    vColor = inColor;
    gl_PointSize = 4.0;
}
"""

POINTS_FRAG = r"""
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
"""

TOPDOWN_FRAG = r"""
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


# ============================================================
# Camera
# ============================================================

class OrbitCamera:
    def __init__(self, target=(0.0, 0.0, 0.0), distance=6.0, yaw=0.8, pitch=0.3):
        self.target = np.array(target, dtype=np.float32)
        self.distance = float(distance)
        self.yaw = float(yaw)
        self.pitch = float(pitch)

        self._dragging = False
        self._last_x = 0.0
        self._last_y = 0.0

        self.rotate_sens = 0.006
        self.pitch_limit = 1.55

    def get_eye(self):
        x = self.distance * math.cos(self.pitch) * math.cos(self.yaw)
        y = self.distance * math.sin(self.pitch)
        z = self.distance * math.cos(self.pitch) * math.sin(self.yaw)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def get_view_matrix(self):
        eye = self.get_eye()
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return pyrr.matrix44.create_look_at(eye, self.target, up, dtype=np.float32)

    def handle_mouse_button(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        if action == glfw.PRESS:
            self._dragging = True
            self._last_x, self._last_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            self._dragging = False

    def handle_cursor_pos(self, window, xpos, ypos):
        if not self._dragging:
            return
        dx = float(xpos - self._last_x)
        dy = float(ypos - self._last_y)
        self._last_x, self._last_y = float(xpos), float(ypos)

        self.yaw += dx * self.rotate_sens
        self.pitch -= dy * self.rotate_sens
        self.pitch = max(-self.pitch_limit, min(self.pitch_limit, self.pitch))

    def handle_scroll(self, window, xoff, yoff):
        self.distance *= math.exp(-float(yoff) * 0.12)
        self.distance = max(0.5, min(2000.0, self.distance))


# ============================================================
# GL helpers
# ============================================================

def _create_fullscreen_vao():
    """Skybox uses aPos attribute; volume/topdown use gl_VertexID with dummy VAO."""
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


def _init_density(sim):
    """WorldStep uses cp.empty for densityfield; seed it so haze is visible."""
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


def _setup_points(sim):
    """Points VBO/VAO for build_point_vertices() -> (N,6) [x,y,z,r,g,b]."""
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


def _make_topdown_image(sim, field: str, z_index: int, mode: str):
    """
    Returns NumPy (NY, NX) float32.
    field: "density" | "flowmag" | "curlmag"
    mode:  "slice" | "max"
    """
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
        img = cp.max(vol, axis=0)     # (NY,NX)
    else:
        img = vol[zi, :, :]           # (NY,NX)

    return img.astype(cp.float32, copy=False).get()


# ============================================================
# Main entry
# ============================================================

def run_viewer(sim):
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Main window
    window = glfw.create_window(1200, 800, "3D Haze (B skybox, P points)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create main window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Shared top-down window (share = main window)
    topwin = glfw.create_window(700, 700, "Top-down (1 density, 2 |flow|, 3 |curl|)", None, window)
    if not topwin:
        glfw.terminate()
        raise RuntimeError("Failed to create top-down window")

    # Camera + callbacks on main window (orbit controls there)
    cam = OrbitCamera()
    glfw.set_mouse_button_callback(window, cam.handle_mouse_button)
    glfw.set_cursor_pos_callback(window, cam.handle_cursor_pos)
    glfw.set_scroll_callback(window, cam.handle_scroll)

    # ------------------------------------------------------------
    # Shared state + key handling
    # ------------------------------------------------------------
    paused = True
    step_once = False
    show_skybox = True
    show_points = True

    top_field = "density"   # density | flowmag | curlmag
    top_mode = "slice"      # slice | max
    top_z = int(sim.NZ) // 2
    top_gain = 1.0

    def on_key(win, key, scancode, action, mods):
        nonlocal paused, step_once, show_skybox, show_points
        nonlocal top_field, top_mode, top_z, top_gain

        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
            glfw.set_window_should_close(topwin, True)

        elif key == glfw.KEY_SPACE:
            paused = not paused
        elif key == glfw.KEY_N:
            step_once = True
        elif key == glfw.KEY_R:
            _init_density(sim)

        elif key == glfw.KEY_B:
            show_skybox = not show_skybox
        elif key == glfw.KEY_P:
            show_points = not show_points

        elif key == glfw.KEY_1:
            top_field = "density"
        elif key == glfw.KEY_2:
            top_field = "flowmag"
        elif key == glfw.KEY_3:
            top_field = "curlmag"

        elif key == glfw.KEY_M:
            top_mode = "max" if top_mode == "slice" else "slice"
        elif key == glfw.KEY_LEFT_BRACKET:
            top_z = max(0, top_z - 1)
        elif key == glfw.KEY_RIGHT_BRACKET:
            top_z = min(int(sim.NZ) - 1, top_z + 1)

        elif key == glfw.KEY_EQUAL:   # '+'
            top_gain *= 1.25
        elif key == glfw.KEY_MINUS:
            top_gain /= 1.25

    # Same key callback on both windows (works regardless of focus)
    glfw.set_key_callback(window, on_key)
    glfw.set_key_callback(topwin, on_key)

    # ------------------------------------------------------------
    # GL resources (create once in main context; shared with topwin)
    # ------------------------------------------------------------
    # Shaders/programs
    sky_prog = compileProgram(
        compileShader(SKYBOX_VERT, gl.GL_VERTEX_SHADER),
        compileShader(SKYBOX_FRAG, gl.GL_FRAGMENT_SHADER),
    )
    vol_prog = compileProgram(
        compileShader(FS_TRI_VERT, gl.GL_VERTEX_SHADER),
        compileShader(VOLUME_FRAG, gl.GL_FRAGMENT_SHADER),
    )
    pts_prog = compileProgram(
        compileShader(POINTS_VERT, gl.GL_VERTEX_SHADER),
        compileShader(POINTS_FRAG, gl.GL_FRAGMENT_SHADER),
    )
    top_prog = compileProgram(
        compileShader(FS_TRI_VERT, gl.GL_VERTEX_SHADER),
        compileShader(TOPDOWN_FRAG, gl.GL_FRAGMENT_SHADER),
    )

    # VAOs
    sky_vao = _create_fullscreen_vao()
    dummy_vao = gl.glGenVertexArrays(1)  # required for gl_VertexID draws in core profile
    pts_vao, pts_vbo, pts_count = _setup_points(sim)
    
    # Create a separate VAO for topwin context (VAOs are not shared between contexts)
    glfw.make_context_current(topwin)
    dummy_vao_topwin = gl.glGenVertexArrays(1)
    glfw.make_context_current(window)  # Switch back to main window context

    # Textures
    nx, ny, nz = int(sim.NX), int(sim.NY), int(sim.NZ)
    dens_tex = _create_density_3d_texture(nx, ny, nz)
    top_tex = _create_scalar_2d_texture(nx, ny)

    # Ensure density starts valid
    _init_density(sim)

    # Volume uniforms
    gl.glUseProgram(vol_prog)
    gl.glUniform1i(gl.glGetUniformLocation(vol_prog, "uDensityTex"), 0)

    loc_inv_vp = gl.glGetUniformLocation(vol_prog, "uInvViewProj")
    loc_cam_pos = gl.glGetUniformLocation(vol_prog, "uCamPos")
    loc_box_min = gl.glGetUniformLocation(vol_prog, "uBoxMin")
    loc_box_max = gl.glGetUniformLocation(vol_prog, "uBoxMax")
    loc_absorp = gl.glGetUniformLocation(vol_prog, "uAbsorption")
    loc_steps = gl.glGetUniformLocation(vol_prog, "uSteps")
    loc_dscale = gl.glGetUniformLocation(vol_prog, "uDensityScale")
    loc_dbias = gl.glGetUniformLocation(vol_prog, "uDensityBias")

    # World-space bounds: match your voxel grid convention [0..NX],[0..NY],[0..NZ]
    box_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    box_max = np.array([float(nx), float(ny), float(nz)], dtype=np.float32)

    gl.glUniform3f(loc_box_min, float(box_min[0]), float(box_min[1]), float(box_min[2]))
    gl.glUniform3f(loc_box_max, float(box_max[0]), float(box_max[1]), float(box_max[2]))

    # Haze tuning (tweak as needed)
    gl.glUniform1f(loc_absorp, 1.8)
    gl.glUniform1i(loc_steps, 140)
    gl.glUniform1f(loc_dscale, 1.0)
    gl.glUniform1f(loc_dbias, 0.0)

    # Points uniforms
    gl.glUseProgram(pts_prog)
    loc_mvp = gl.glGetUniformLocation(pts_prog, "uMVP")

    # Topdown uniforms
    gl.glUseProgram(top_prog)
    gl.glUniform1i(gl.glGetUniformLocation(top_prog, "uTex"), 0)
    loc_gain = gl.glGetUniformLocation(top_prog, "uGain")

    # GL state
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    # Frame volume with camera
    center = 0.5 * (box_min + box_max)
    extent = (box_max - box_min)
    radius = 0.5 * float(np.linalg.norm(extent))
    cam.target = center.astype(np.float32)
    cam.distance = max(3.0, radius * 2.2)

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
    while (not glfw.window_should_close(window)) and (not glfw.window_should_close(topwin)):
        glfw.poll_events()

        # Step sim
        if (not paused) or step_once:
            # Prefer density-only if available
            if hasattr(sim, "step_densityfield"):
                sim.step_densityfield(0.1)
            else:
                sim.step(0.1)
            step_once = False

        # Upload density every frame (both windows use it)
        _upload_density_3d(dens_tex, sim.densityfield)

        # Upload points if enabled
        if show_points and hasattr(sim, "build_point_vertices"):
            _upload_points(sim, pts_vbo)

        # Build matrices for main window
        glfw.make_context_current(window)
        w, h = glfw.get_framebuffer_size(window)
        aspect = w / max(1, h)
        proj = pyrr.matrix44.create_perspective_projection(45.0, aspect, 0.1, 1000.0, dtype=np.float32)
        view = cam.get_view_matrix()
        vp = proj @ view

        inv_vp_vol = np.linalg.inv(vp).astype(np.float32, copy=False)
        inv_vp_sky = inv_vp_vol.copy()
        inv_vp_sky[:3, 3] = 0.0

        inv_view = np.linalg.inv(view).astype(np.float32, copy=False)
        cam_pos4 = inv_view @ np.array([0, 0, 0, 1], dtype=np.float32)
        cam_pos = cam_pos4[:3] / max(cam_pos4[3], 1e-6)

        # -------------------
        # Render main window
        # -------------------
        gl.glViewport(0, 0, w, h)
        gl.glClearColor(0.02, 0.02, 0.04, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Skybox
        if show_skybox:
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glUseProgram(sky_prog)
            loc_sky = gl.glGetUniformLocation(sky_prog, "uInvViewProj")
            gl.glUniformMatrix4fv(loc_sky, 1, gl.GL_FALSE, inv_vp_sky)
            gl.glBindVertexArray(sky_vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            gl.glBindVertexArray(0)

        # Volume haze
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(vol_prog)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_3D, dens_tex)
        gl.glUniformMatrix4fv(loc_inv_vp, 1, gl.GL_FALSE, inv_vp_vol)
        gl.glUniform3f(loc_cam_pos, float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))

        gl.glBindVertexArray(dummy_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)

        # Points overlay
        if show_points and hasattr(sim, "build_point_vertices"):
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glUseProgram(pts_prog)
            gl.glUniformMatrix4fv(loc_mvp, 1, gl.GL_FALSE, vp.astype(np.float32, copy=False))
            gl.glBindVertexArray(pts_vao)
            gl.glDrawArrays(gl.GL_POINTS, 0, pts_count)
            gl.glBindVertexArray(0)

        glfw.swap_buffers(window)

        # -------------------
        # Render top-down window
        # -------------------
        glfw.make_context_current(topwin)
        tw, th = glfw.get_framebuffer_size(topwin)
        gl.glViewport(0, 0, tw, th)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Build and upload the topdown scalar
        img = _make_topdown_image(sim, top_field, top_z, top_mode)  # (NY,NX)
        _upload_scalar_2d(top_tex, img)

        gl.glUseProgram(top_prog)
        gl.glUniform1f(loc_gain, float(top_gain))
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, top_tex)

        gl.glBindVertexArray(dummy_vao_topwin)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        glfw.swap_buffers(topwin)

    glfw.terminate()
