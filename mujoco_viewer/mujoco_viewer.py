import mujoco
import glfw
import sys
from threading import Lock
import numpy as np
import time
import imageio


class MujocoViewer:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._paused = False
        self._transparent = False
        self._contacts = False
        self._render_every_frame = True
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._hide_menu = False

        # glfw init
        glfw.init()
        width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        self.window = glfw.create_window(
            width // 2, height // 2, "mujoco", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
            self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width

        # set callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(
            self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        # create options, camera, scene, context
        self.vopt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # get viewport
        self.viewport = mujoco.MjrRect(
            0, 0, framebuffer_width, framebuffer_height)

        # overlay, markers
        self._overlay = {}
        self._markers = []

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return
        # Switch cameras
        elif key == glfw.KEY_TAB:
            self.cam.fixedcamid += 1
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            if self.cam.fixedcamid >= self.model.ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Pause simulation
        elif key == glfw.KEY_SPACE and self._paused is not None:
            self._paused = not self._paused
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        # Slows down simulation
        elif key == glfw.KEY_S:
            self._run_speed /= 2.0
        # Speeds up simulation
        elif key == glfw.KEY_F:
            self._run_speed *= 2.0
        # Turn off / turn on rendering every frame.
        elif key == glfw.KEY_D:
            self._render_every_frame = not self._render_every_frame
        # Capture screenshot
        elif key == glfw.KEY_T:
            img = np.zeros(
                (glfw.get_framebuffer_size(
                    self.window)[1], glfw.get_framebuffer_size(
                    self.window)[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, self.viewport, self.ctx)
            imageio.imwrite(self._image_path % self._image_idx, np.flipud(img))
            self._image_idx += 1
        # Display contact forces
        elif key == glfw.KEY_C:
            self._contacts = not self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._contacts
        # Display coordinate frames
        elif key == glfw.KEY_E:
            self.vopt.frame = 1 - self.vopt.frame
        # Hide overlay menu
        elif key == glfw.KEY_H:
            self._hide_menu = not self._hide_menu
        # Make transparent
        elif key == glfw.KEY_R:
            self._transparent = not self._transparent
            if self._transparent:
                self.model.geom_rgba[:, 3] /= 5.0
            else:
                self.model.geom_rgba[:, 3] *= 5.0
        # Geom group visibility
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        # Quit
        if key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            print("Quitting.")
            glfw.terminate()
            sys.exit(0)

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model,
                action,
                dx / height,
                dy / height,
                self.scn,
                self.cam)

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self._button_right_pressed = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.scn, self.cam)

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(
                'Ran out of geoms. maxgeom: %d' %
                self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)))
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        self.scn.ngeom += 1

        return

    def _create_overlay(self):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

        def add_overlay(gridpos, text1, text2):
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1 + "\n"
            self._overlay[gridpos][1] += text2 + "\n"

        if self._render_every_frame:
            add_overlay(topleft, "", "")
        else:
            add_overlay(
                topleft,
                "Run speed = %.3f x real time" %
                self._run_speed,
                "[S]lower, [F]aster")
        add_overlay(
            topleft,
            "Ren[d]er every frame",
            "On" if self._render_every_frame else "Off")
        add_overlay(
            topleft, "Switch camera (#cams = %d)" %
            (self.model.ncam + 1), "[Tab] (camera ID = %d)" %
            self.cam.fixedcamid)
        add_overlay(
            topleft,
            "[C]ontact forces",
            "On" if self._contacts else "Off")
        add_overlay(
            topleft,
            "T[r]ansparent",
            "On" if self._transparent else "Off")
        if self._paused is not None:
            if not self._paused:
                add_overlay(topleft, "Stop", "[Space]")
            else:
                add_overlay(topleft, "Start", "[Space]")
                add_overlay(
                    topleft,
                    "Advance simulation by one step",
                    "[right arrow]")
        add_overlay(
            topleft,
            "Referenc[e] frames",
            "On" if self.vopt.frame == 1 else "Off")
        add_overlay(topleft, "[H]ide Menu", "")
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            add_overlay(topleft, "Cap[t]ure frame", "Saved as %s" % fname)
        else:
            add_overlay(topleft, "Cap[t]ure frame", "")
        add_overlay(topleft, "Toggle geomgroup visibility", "0-4")

        add_overlay(
            bottomleft, "FPS", "%d%s" %
            (1 / self._time_per_render, ""))
        add_overlay(
            bottomleft, "Solver iterations", str(
                self.data.solver_iter + 1))
        add_overlay(
            bottomleft, "Step", str(
                round(
                    self.data.time / self.model.opt.timestep)))
        add_overlay(bottomleft, "timestep", "%.5f" % self.model.opt.timestep)

    def render(self):
        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.terminate()
                sys.exit(0)
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window)
            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    mujoco.MjvPerturb(),
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn)
                # marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # render
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)
                # overlay items
                if not self._hide_menu:
                    for gridpos, [t1, t2] in self._overlay.items():
                        mujoco.mjr_overlay(
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            gridpos,
                            self.viewport,
                            t1,
                            t2,
                            self.ctx)
                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + \
                0.1 * (time.time() - render_start)

            # clear overlay
            self._overlay.clear()

        if self._paused:
            while self._paused:
                update()
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / \
                (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        # clear markers
        self._markers[:] = []

    def close(self):
        glfw.terminate()
        sys.exit(0)
