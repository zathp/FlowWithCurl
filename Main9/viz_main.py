"""Modular visualizer runner that composes the helper modules."""
import glfw
import numpy as np
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr

from viz_shaders import (
    SKYBOX_VERT, SKYBOX_FRAG, FS_TRI_VERT, VOLUME_FRAG,
    POINTS_VERT, POINTS_FRAG, TOPDOWN_FRAG,
)
from viz_camera import OrbitCamera
from viz_gl_helpers import (
    _create_fullscreen_vao, _create_density_3d_texture, _upload_density_3d,
    _create_scalar_2d_texture, _upload_scalar_2d, _setup_points, _upload_points,
)
from viz_topdown import _init_density, _make_topdown_image


def run_viewer(sim):
    if not glfw.init():
        raise RuntimeError("Failed to init GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(1200, 800, "3D Haze (B skybox, P points)", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create main window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    topwin = glfw.create_window(700, 700, "Top-down (1 density, 2 |flow|, 3 |curl|)", None, window)
    if not topwin:
        glfw.terminate()
        raise RuntimeError("Failed to create top-down window")

    cam = OrbitCamera()
    glfw.set_mouse_button_callback(window, cam.handle_mouse_button)
    glfw.set_cursor_pos_callback(window, cam.handle_cursor_pos)
    glfw.set_scroll_callback(window, cam.handle_scroll)

    paused = True
    step_once = False
    show_skybox = True
    show_points = True

    top_field = "density"
    top_mode = "slice"
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
        elif key == glfw.KEY_EQUAL:
            top_gain *= 1.25
        elif key == glfw.KEY_MINUS:
            top_gain /= 1.25

    glfw.set_key_callback(window, on_key)
    glfw.set_key_callback(topwin, on_key)

    sky_prog = compileProgram(compileShader(SKYBOX_VERT, gl.GL_VERTEX_SHADER), compileShader(SKYBOX_FRAG, gl.GL_FRAGMENT_SHADER))
    vol_prog = compileProgram(compileShader(FS_TRI_VERT, gl.GL_VERTEX_SHADER), compileShader(VOLUME_FRAG, gl.GL_FRAGMENT_SHADER))
    pts_prog = compileProgram(compileShader(POINTS_VERT, gl.GL_VERTEX_SHADER), compileShader(POINTS_FRAG, gl.GL_FRAGMENT_SHADER))
    top_prog = compileProgram(compileShader(FS_TRI_VERT, gl.GL_VERTEX_SHADER), compileShader(TOPDOWN_FRAG, gl.GL_FRAGMENT_SHADER))

    sky_vao = _create_fullscreen_vao()
    dummy_vao = gl.glGenVertexArrays(1)
    pts_vao, pts_vbo, pts_count = _setup_points(sim)

    glfw.make_context_current(topwin)
    dummy_vao_topwin = gl.glGenVertexArrays(1)
    glfw.make_context_current(window)

    nx, ny, nz = int(sim.NX), int(sim.NY), int(sim.NZ)
    dens_tex = _create_density_3d_texture(nx, ny, nz)
    top_tex = _create_scalar_2d_texture(nx, ny)

    _init_density(sim)

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

    box_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    box_max = np.array([float(nx), float(ny), float(nz)], dtype=np.float32)

    gl.glUniform3f(loc_box_min, float(box_min[0]), float(box_min[1]), float(box_min[2]))
    gl.glUniform3f(loc_box_max, float(box_max[0]), float(box_max[1]), float(box_max[2]))

    gl.glUniform1f(loc_absorp, 1.8)
    gl.glUniform1i(loc_steps, 140)
    gl.glUniform1f(loc_dscale, 1.0)
    gl.glUniform1f(loc_dbias, 0.0)

    gl.glUseProgram(pts_prog)
    loc_mvp = gl.glGetUniformLocation(pts_prog, "uMVP")

    gl.glUseProgram(top_prog)
    gl.glUniform1i(gl.glGetUniformLocation(top_prog, "uTex"), 0)
    loc_gain = gl.glGetUniformLocation(top_prog, "uGain")

    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    center = 0.5 * (box_min + box_max)
    extent = (box_max - box_min)
    radius = 0.5 * float(np.linalg.norm(extent))
    cam.target = center.astype(np.float32)
    cam.distance = max(3.0, radius * 2.2)

    while (not glfw.window_should_close(window)) and (not glfw.window_should_close(topwin)):
        glfw.poll_events()

        if (not paused) or step_once:
            if hasattr(sim, "step_densityfield"):
                sim.step_densityfield(0.1)
            else:
                sim.step(0.1)
            step_once = False

        _upload_density_3d(dens_tex, sim.densityfield)

        if show_points and hasattr(sim, "build_point_vertices"):
            _upload_points(sim, pts_vbo)

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

        gl.glViewport(0, 0, w, h)
        gl.glClearColor(0.02, 0.02, 0.04, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if show_skybox:
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glUseProgram(sky_prog)
            loc_sky = gl.glGetUniformLocation(sky_prog, "uInvViewProj")
            gl.glUniformMatrix4fv(loc_sky, 1, gl.GL_FALSE, inv_vp_sky)
            gl.glBindVertexArray(sky_vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            gl.glBindVertexArray(0)

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

        if show_points and hasattr(sim, "build_point_vertices"):
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glUseProgram(pts_prog)
            gl.glUniformMatrix4fv(loc_mvp, 1, gl.GL_FALSE, vp.astype(np.float32, copy=False))
            gl.glBindVertexArray(pts_vao)
            gl.glDrawArrays(gl.GL_POINTS, 0, pts_count)
            gl.glBindVertexArray(0)

        glfw.swap_buffers(window)

        glfw.make_context_current(topwin)
        tw, th = glfw.get_framebuffer_size(topwin)
        gl.glViewport(0, 0, tw, th)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        img = _make_topdown_image(sim, top_field, top_z, top_mode)
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
