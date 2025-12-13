# orbit_camera.py
import math
import numpy as np
import glfw
import pyrr

class OrbitCamera:
    def __init__(self):
        self.target   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.distance = 5.0
        self.yaw      = 0.0     # radians
        self.pitch    = 0.5     # radians

        self.dragging = False
        self.last_x   = 0.0
        self.last_y   = 0.0

        # tunables
        self.orbit_sensitivity = 0.005
        self.zoom_sensitivity  = 1.0
        self.min_distance      = 1.0
        self.max_distance      = 50.0

    # ---------- math helpers ----------


    def get_view_matrix(self):
        x = self.distance * math.cos(self.pitch) * math.cos(self.yaw)
        y = self.distance * math.sin(self.pitch)
        z = self.distance * math.cos(self.pitch) * math.sin(self.yaw)

        eye = self.target + np.array([x, y, z], dtype=np.float32)
        up  = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        return pyrr.matrix44.create_look_at(eye, self.target, up, dtype=np.float32)


    # ---------- GLFW callback handlers ----------

    def handle_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.dragging = True
                self.last_x, self.last_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                self.dragging = False

    def handle_cursor_pos(self, window, xpos, ypos):
        if not self.dragging:
            return

        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x = xpos
        self.last_y = ypos

        self.yaw   += dx * self.orbit_sensitivity
        self.pitch += -dy * self.orbit_sensitivity  # invert Y for natural feel

        # clamp pitch to avoid flipping over
        pitch_limit = 1.5  # ~86 degrees
        if self.pitch >  pitch_limit: self.pitch =  pitch_limit
        if self.pitch < -pitch_limit: self.pitch = -pitch_limit

    def handle_scroll(self, window, xoffset, yoffset):
        self.distance -= yoffset * self.zoom_sensitivity
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))


# ---------- minimal usage example ----------

if __name__ == "__main__":
    if not glfw.init():
        raise SystemExit("Failed to init GLFW")

    window = glfw.create_window(800, 600, "Orbit Camera Test", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Failed to create window")

    glfw.make_context_current(window)

    cam = OrbitCamera

    while not glfw.window_should_close(window):
        glfw.poll_events()

        view = cam.get_view_matrix()
        print(view)
        # send `view` to your shader as a 4x4 matrix (column-major):
        # glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

        # ... your drawing code here ...

        glfw.swap_buffers(window)

    glfw.terminate()

import pyrr

prevtime = 0

def getviev(now, cam: OrbitCamera):
    radius = 8.0
    eye = np.array([
        radius * math.cos(0 * 0.3),
        3.0,
        radius * math.sin(0 * 0.3)
    ], dtype=np.float32)
    center = np.zeros(3, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)

    matOrb1 = cam.get_view_matrix()
    matOrb2 = pyrr.matrix44.create_look_at(eye, center, up, dtype=np.float32)
    return matOrb1 @ matOrb2