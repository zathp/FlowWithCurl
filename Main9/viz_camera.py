import math
import numpy as np
import glfw
import pyrr


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
