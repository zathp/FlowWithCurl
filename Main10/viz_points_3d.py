import glfw
import numpy as np
import cupy as cp
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
import tempfile
import os

POINTS_VERT = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color_in;
layout(location = 2) in float size_in;
uniform mat4 mvp;
uniform float min_size;
out vec3 color;
void main() {
    gl_Position = mvp * vec4(position, 1.0);
    color = color_in;
    gl_PointSize = max(size_in, min_size);
}
"""

POINTS_FRAG = """
#version 330 core
in vec3 color;
out vec4 out_col;
void main() {
    out_col = vec4(color, 1.0);
}
"""


def _create_shader():
    return compileProgram(compileShader(POINTS_VERT, gl.GL_VERTEX_SHADER), compileShader(POINTS_FRAG, gl.GL_FRAGMENT_SHADER))


def run_viewer(sim, width=1000, height=800):
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(width, height, "3D Points + Bounding Box", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    shader = _create_shader()
    gl.glUseProgram(shader)

    # Create buffers
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    # allocate initial empty buffer (will update each frame)
    max_points = sim.NX * sim.NY * sim.NZ
    float_size = 4
    vertex_stride = 7 * float_size  # position(3) + color(3) + size(1)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, max_points * vertex_stride, None, gl.GL_DYNAMIC_DRAW)

    # position attribute
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, gl.ctypes.c_void_p(0))
    # color attribute
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, gl.ctypes.c_void_p(12))
    # size attribute
    gl.glEnableVertexAttribArray(2)
    gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, gl.ctypes.c_void_p(24))

    # Bounding box line setup (12 edges * 2 vertices)
    box_vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(box_vao)
    box_vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, box_vbo)

    # simple color for lines
    line_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # enable depth test
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    paused = True
    step_once = False
    sample_once = False
    camera_radius = 1.0
    render_mode = "points"  # or "vectors"
    vector_field_choice = "flow"  # 'flow' or 'curl'

    def key_cb(window, key, scancode, action, mods):
        nonlocal paused, step_once, sample_once, camera_radius, center, yaw, pitch, up, render_mode, vector_field_choice
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                paused = not paused
            elif key == glfw.KEY_N:
                step_once = True
            elif key == glfw.KEY_P:
                # request a random sample print on next frame
                sample_once = True
            elif key == glfw.KEY_KP_ADD or key == glfw.KEY_EQUAL:
                camera_radius *= 0.85
            elif key == glfw.KEY_KP_SUBTRACT or key == glfw.KEY_MINUS:
                camera_radius *= 1.15
            elif key == glfw.KEY_R:
                # reset camera to initial framing
                camera_radius = float(radius)
                center[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                # recompute yaw/pitch to defaults
                try:
                    yaw = initial_yaw
                    pitch = initial_pitch
                except Exception:
                    pass
            elif key == glfw.KEY_V:
                # toggle render mode between points and vectors
                if render_mode == "points":
                    render_mode = "vectors"
                else:
                    render_mode = "points"
                print("Render mode:", render_mode)
            elif key == glfw.KEY_B:
                # toggle which vector field to view
                if vector_field_choice == "flow":
                    vector_field_choice = "curl"
                else:
                    vector_field_choice = "flow"
                print("Vector field:", vector_field_choice)
            # keyboard movement: j=strafe left, l=strafe right, i=forward, k=back
            elif key == glfw.KEY_J:
                # strafe left
                dir_vec = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], dtype=np.float32)
                eye_pos = center + dir_vec * camera_radius
                forward = center - eye_pos
                forward = forward / (np.linalg.norm(forward) + 1e-9)
                right = np.cross(forward, up)
                right = right / (np.linalg.norm(right) + 1e-9)
                scale = camera_radius * 0.02
                center += (-1.0 * scale) * right
                center = np.minimum(np.maximum(center, cam_bounds_min), cam_bounds_max)
            elif key == glfw.KEY_L:
                # strafe right
                dir_vec = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], dtype=np.float32)
                eye_pos = center + dir_vec * camera_radius
                forward = center - eye_pos
                forward = forward / (np.linalg.norm(forward) + 1e-9)
                right = np.cross(forward, up)
                right = right / (np.linalg.norm(right) + 1e-9)
                scale = camera_radius * 0.02
                center += (1.0 * scale) * right
                center = np.minimum(np.maximum(center, cam_bounds_min), cam_bounds_max)
            elif key == glfw.KEY_I:
                # move forward along view direction
                dir_vec = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], dtype=np.float32)
                forward_dir = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)
                scale = camera_radius * 0.05
                center += forward_dir * scale
                center = np.minimum(np.maximum(center, cam_bounds_min), cam_bounds_max)
            elif key == glfw.KEY_K:
                # move backward along view direction
                dir_vec = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], dtype=np.float32)
                forward_dir = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)
                scale = camera_radius * 0.05
                center -= forward_dir * scale
                center = np.minimum(np.maximum(center, cam_bounds_min), cam_bounds_max)
            # keyboard vertical movement: h = up, n = down
            elif key == glfw.KEY_H:
                # move up along world up vector
                scale = camera_radius * 0.02
                center += up * scale
                center = np.minimum(np.maximum(center, cam_bounds_min), cam_bounds_max)
            elif key == glfw.KEY_N:
                scale = camera_radius * 0.02
                center -= up * scale
                center = np.minimum(np.maximum(center, cam_bounds_min), cam_bounds_max)
            elif key == glfw.KEY_D:
                # dump current sim state to temp file for plotting
                try:
                    dump_path = os.path.join(tempfile.gettempdir(), 'flowstep_dump.npz')
                    flow_np = cp.asnumpy(sim.flowfield) if isinstance(sim.flowfield, cp.ndarray) else np.asarray(sim.flowfield)
                    curl_np = cp.asnumpy(sim.curlfield) if isinstance(sim.curlfield, cp.ndarray) else np.asarray(sim.curlfield)
                    density_np = cp.asnumpy(sim.densityfield) if isinstance(sim.densityfield, cp.ndarray) else np.asarray(sim.densityfield)
                    np.savez(dump_path, flowfield=flow_np, curlfield=curl_np, densityfield=density_np,
                             NX=sim.NX, NY=sim.NY, NZ=sim.NZ, LX=sim.LX, LY=sim.LY, LZ=sim.LZ)
                    print(f"Dumped to {dump_path}")
                except Exception as e:
                    print(f"Error dumping: {e}")
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
        # clamp camera radius to allowed range
        camera_radius = float(np.clip(camera_radius, cam_radius_min, cam_radius_max))

    glfw.set_key_callback(window, key_cb)

    def scroll_cb(window, xoffset, yoffset):
        nonlocal camera_radius
        # positive yoffset -> scroll up -> zoom in
        try:
            camera_radius *= (0.9 ** float(yoffset))
        except Exception:
            camera_radius *= (0.9 if yoffset > 0 else 1.1)

    # Camera bounding box (allows camera center to move outside particle box)
    cam_bounds_min = np.array([-200.0, -200.0, -50.0], dtype=np.float32)
    cam_bounds_max = np.array([200.0, 200.0, 50.0], dtype=np.float32)
    # camera radius limits
    cam_radius_min = 1.0
    cam_radius_max = 2000.0

    # mouse / camera interaction state
    yaw = 0.0
    pitch = 0.0
    rotating = False
    panning = False
    last_x = 0.0
    last_y = 0.0

    def mouse_button_cb(window, button, action, mods):
        nonlocal rotating, panning, last_x, last_y
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                rotating = True
                last_x, last_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                rotating = False
        if button == glfw.MOUSE_BUTTON_MIDDLE or button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                panning = True
                last_x, last_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                panning = False

    def cursor_pos_cb(window, x, y):
        nonlocal last_x, last_y, yaw, pitch, center, camera_radius
        dx = x - last_x
        dy = y - last_y
        last_x, last_y = x, y
        if rotating:
            # rotate: change yaw and pitch
            yaw += dx * 0.005
            pitch += -dy * 0.005
            # clamp pitch to avoid flip
            limit = 1.49
            if pitch > limit:
                pitch = limit
            if pitch < -limit:
                pitch = -limit
        elif panning:
            # pan: move the center in camera plane
            # compute current eye and basis
            dir_vec = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], dtype=np.float32)
            eye_pos = center + dir_vec * camera_radius
            forward = center - eye_pos
            forward = forward / (np.linalg.norm(forward) + 1e-9)
            right = np.cross(forward, up)
            right = right / (np.linalg.norm(right) + 1e-9)
            up_cam = np.cross(right, forward)
            # scale movement by radius
            scale = camera_radius * 0.002
            center += (-dx * scale) * right + (dy * scale) * up_cam
            # clamp camera center to camera bounding box
            center = np.minimum(np.maximum(center, cam_bounds_min), cam_bounds_max)

    # register callbacks after definitions
    glfw.set_scroll_callback(window, scroll_cb)
    glfw.set_mouse_button_callback(window, mouse_button_cb)
    glfw.set_cursor_pos_callback(window, cursor_pos_cb)

    def build_vector_vertices(sim, field_name="flow", stride=4, scale=1.0):
        """Build line segment endpoints for sampled vector field.

        Returns a numpy array of shape (M,6) with [x,y,z,r,g,b] for each vertex.
        Each vector becomes two vertices (start, end).
        """
        # choose field
        if field_name == "flow":
            field_cp = sim.flowfield
        else:
            field_cp = sim.curlfield

        # copy to host
        try:
            field = cp.asnumpy(field_cp)
        except Exception:
            field = np.asarray(field_cp)

        nz, ny, nx, _ = field.shape
        # grid centers
        ox = -sim.LX * (sim.NX - 1) / 2.0
        oy = -sim.LY * (sim.NY - 1) / 2.0
        oz = -sim.LZ * (sim.NZ - 1) / 2.0

        idxs = []
        for iz in range(0, nz, stride):
            for iy in range(0, ny, stride):
                for ix in range(0, nx, stride):
                    idxs.append((iz, iy, ix))

        out = np.empty((len(idxs) * 2, 7), dtype=np.float32)
        vi = 0
        # scale factor relative to cell size
        cell_scale = max(sim.LX, sim.LY, sim.LZ)
        for (iz, iy, ix) in idxs:
            cx = ox + ix * sim.LX
            cy = oy + iy * sim.LY
            cz = oz + iz * sim.LZ
            vec = field[iz, iy, ix]
            mag = np.linalg.norm(vec) + 1e-9
            dirv = vec / mag
            end = np.array([cx, cy, cz], dtype=np.float32) + dirv.astype(np.float32) * (scale * cell_scale * 0.5)
            # color by direction (normalized to 0..1)
            color = (np.abs(dirv) * 0.5 + 0.25).astype(np.float32)
            out[vi, 0:3] = [cx, cy, cz]
            out[vi, 3:6] = color
            out[vi, 6] = 1.0  # size placeholder
            vi += 1
            out[vi, 0:3] = end
            out[vi, 3:6] = color
            out[vi, 6] = 1.0
            vi += 1
        return out

    # camera setup (fixed orbit)
    # Use the requested fixed bounding box: (-50,-50,-5) .. (50,50,5)
    x_min = -50.0
    x_max = 50.0
    y_min = -50.0
    y_max = 50.0
    z_min = -5.0
    z_max = 5.0

    # Position camera outside the particle bounding box looking at center
    # Camera positioned at a comfortable distance to view the entire box
    radius = 150.0
    # Start camera in front and above the box (positive x, y, z all outside box bounds)
    eye = np.array([120.0, 80.0, 60.0], dtype=np.float32)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Use perspective projection
    fov = 60.0  # field of view in degrees
    aspect = width / height
    near = 1.0
    far = 5000.0
    proj = pyrr.matrix44.create_perspective_projection_matrix(fov, aspect, near, far, dtype=np.float32)
    # initialize camera radius and derive initial yaw/pitch from eye
    camera_radius = float(radius)
    dir_vec_init = eye - center
    dir_len = np.linalg.norm(dir_vec_init) + 1e-9
    dir_unit = dir_vec_init / dir_len
    initial_pitch = float(np.arcsin(dir_unit[2]))
    initial_yaw = float(np.arctan2(dir_unit[1], dir_unit[0]))
    yaw = initial_yaw
    pitch = initial_pitch
    view = pyrr.matrix44.create_look_at(eye, center, up, dtype=np.float32)

    mvp_loc = gl.glGetUniformLocation(shader, "mvp")
    min_size_loc = gl.glGetUniformLocation(shader, "min_size")

    # Precompute bounding box edges (use fixed coordinates)
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
    ], dtype=np.float32)

    # edges as index pairs
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    line_verts = np.array([corners[a] for a,b in edges for a in (a,)], dtype=np.float32) # will replace below
    # build line vertex array as sequence of pairs
    line_verts = np.empty((len(edges)*2,3), dtype=np.float32)
    idx = 0
    for a,b in edges:
        line_verts[idx] = corners[a]; idx += 1
        line_verts[idx] = corners[b]; idx += 1

    # upload static line VBO
    gl.glBindVertexArray(box_vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, box_vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, line_verts.nbytes, line_verts, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 12, gl.ctypes.c_void_p(0))

    frame = 0
    vbo_size = max_points * vertex_stride  # track current VBO allocation
    # rendering loop
    while not glfw.window_should_close(window):
        w, h = glfw.get_framebuffer_size(window)
        gl.glViewport(0, 0, w, h)
        gl.glClearColor(0.1, 0.1, 0.12, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if (not paused) or step_once:
            sim.step(0.05)
            step_once = False

        # upload either particle vertices or vector vertices depending on render mode
        if render_mode == "points":
            verts_cp = sim.build_point_vertices()  # cupy array (N,6)
            verts = np.asarray(cp.asnumpy(verts_cp), dtype=np.float32)
        else:
            # build vector lines (sampled)
            # choose stride so we don't overload the GPU
            stride = max(1, sim.NX // 20)
            verts = build_vector_vertices(sim, field_name=vector_field_choice, stride=stride, scale=1.0)
        # update VBO - reallocate if needed
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        size = verts.nbytes
        if size > vbo_size:
            # need to reallocate
            gl.glBufferData(gl.GL_ARRAY_BUFFER, size, verts, gl.GL_DYNAMIC_DRAW)
            vbo_size = size
        else:
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, size, verts)

        # compute MVP (update eye from yaw/pitch and camera_radius)
        dir_vec = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], dtype=np.float32)
        eye = center + dir_vec * camera_radius
        view = pyrr.matrix44.create_look_at(eye, center, up, dtype=np.float32)
        mvp = proj @ view
        # OpenGL expects column-major matrix data; pyrr returns row-major arrays.
        mvp_upload = mvp.T.astype(np.float32)
        gl.glUseProgram(shader)
        gl.glUniformMatrix4fv(mvp_loc, 1, gl.GL_FALSE, mvp_upload)
        gl.glUniform1f(min_size_loc, 3.0)

        # draw depending on mode
        gl.glBindVertexArray(vao)
        # If user requested a random sample, print one particle's position and color
        if sample_once:
            sample_once = False
            try:
                if verts.shape[0] > 0:
                    idx = int(np.random.randint(0, verts.shape[0]))
                    pos = verts[idx, :3]
                    col = verts[idx, 3:6]
                    print(f"Sample idx={idx} pos={pos} color={col}")
                else:
                    print("No vertices to sample")
            except Exception as e:
                print("Error sampling vertex:", e)
        # debug: print vertex count and bounds occasionally and NDC ranges
        if frame % 60 == 0:
            try:
                pmin = verts[:, :3].min(axis=0)
                pmax = verts[:, :3].max(axis=0)
                # compute clip/NDC coords for a quick frustum check
                hom = np.hstack((verts[:, :3], np.ones((verts.shape[0], 1), dtype=np.float32)))
                # treat vertices as row vectors and multiply by mvp^T to match upload order
                clip = hom @ mvp.T
                ndc = clip[:, :3] / clip[:, 3:4]
                ndc_min = ndc.min(axis=0)
                ndc_max = ndc.max(axis=0)
                print(f"Frame {frame}: points={verts.shape[0]} pos_min={pmin} pos_max={pmax} ndc_min={ndc_min} ndc_max={ndc_max}")
            except Exception:
                print(f"Frame {frame}: points={verts.shape[0]} (no position stats)")
        frame += 1

        if render_mode == "points":
            gl.glDrawArrays(gl.GL_POINTS, 0, verts.shape[0])
        else:
            # vectors drawn as lines (each vector is two vertices)
            gl.glDrawArrays(gl.GL_LINES, 0, verts.shape[0])

        # draw bounding box lines
        gl.glUseProgram(shader)
        gl.glBindVertexArray(box_vao)
        # set a constant color via client-side attribute: create small color buffer state by disabling attrib1 and using constant via vertex attrib
        gl.glDisableVertexAttribArray(1)
        # set color via vertex attribute binding to constant
        gl.glVertexAttrib3f(1, 1.0, 1.0, 1.0)
        gl.glDrawArrays(gl.GL_LINES, 0, line_verts.shape[0])
        gl.glEnableVertexAttribArray(1)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()
