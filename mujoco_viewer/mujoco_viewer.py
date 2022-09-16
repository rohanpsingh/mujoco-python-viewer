import mujoco
import glfw
import numpy as np
import time
import pathlib
import yaml
from .callbacks import Callbacks


class MujocoViewer(Callbacks):
    def __init__(
            self,
            model,
            data,
            mode='window',
            title="mujoco-python-viewer",
            width=None,
            height=None,
            hide_menus=False):
        super().__init__(hide_menus)

        self.model = model
        self.data = data
        self.render_mode = mode
        if self.render_mode not in ['offscreen', 'window']:
            raise NotImplementedError(
                "Invalid mode. Only 'offscreen' and 'window' are supported.")

        # keep true while running
        self.is_alive = True

        self.CONFIG_PATH = pathlib.Path.joinpath(
            pathlib.Path.home(), ".config/mujoco_viewer/config.yaml")

        # glfw init
        glfw.init()

        if not width:
            width, _ = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if not height:
            _, height = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if self.render_mode == 'offscreen':
            glfw.window_hint(glfw.VISIBLE, 0)

        self.window = glfw.create_window(
            width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
            self.window)

        # install callbacks only for 'window' mode
        if self.render_mode == 'window':
            window_width, _ = glfw.get_window_size(self.window)
            self._scale = framebuffer_width * 1.0 / window_width

            # set callbacks
            glfw.set_cursor_pos_callback(
                self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(
                self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_key_callback(self.window, self._key_callback)

        # create options, camera, scene, context
        self.vopt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.pert = mujoco.MjvPerturb()
        self.ctx = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # load camera from configuration (if available)
        pathlib.Path(
            self.CONFIG_PATH.parent).mkdir(
            parents=True,
            exist_ok=True)
        pathlib.Path(self.CONFIG_PATH).touch(exist_ok=True)
        with open(self.CONFIG_PATH, "r") as f:
            try:
                cam_config = {
                    "type": self.cam.type,
                    "fixedcamid": self.cam.fixedcamid,
                    "trackbodyid": self.cam.trackbodyid,
                    "lookat": self.cam.lookat.tolist(),
                    "distance": self.cam.distance,
                    "azimuth": self.cam.azimuth,
                    "elevation": self.cam.elevation
                }
                load_config = yaml.safe_load(f)
                if isinstance(load_config, dict):
                    for key, val in load_config.items():
                        if key in cam_config.keys():
                            cam_config[key] = val
                if cam_config["type"] == mujoco.mjtCamera.mjCAMERA_FIXED:
                    if cam_config["fixedcamid"] < self.model.ncam:
                        self.cam.type = cam_config["type"]
                        self.cam.fixedcamid = cam_config["fixedcamid"]
                if cam_config["type"] == mujoco.mjtCamera.mjCAMERA_TRACKING:
                    if cam_config["trackbodyid"] < self.model.nbody:
                        self.cam.type = cam_config["type"]
                        self.cam.trackbodyid = cam_config["trackbodyid"]
                self.cam.lookat = np.array(cam_config["lookat"])
                self.cam.distance = cam_config["distance"]
                self.cam.azimuth = cam_config["azimuth"]
                self.cam.elevation = cam_config["elevation"]
            except yaml.YAMLError as e:
                print(e)

        # get viewport
        self.viewport = mujoco.MjrRect(
            0, 0, framebuffer_width, framebuffer_height)

        # overlay, markers
        self._overlay = {}
        self._markers = []
        
    def show_actuator_forces(
        self,
        site_list,
        actuator_list,
        rgba_list=[1, 0, 1, 1],
        force_scale=0.05,
        arrow_radius=0.03,
        show_force_labels=False,
    ) -> None:
        if show_force_labels is False:
            for i in range(0, len(site_list)):
                self.add_marker(
                    pos=self.data.site(i).xpos,
                    mat=self.data.site(i).xmat,
                    size=[
                        arrow_radius,
                        arrow_radius,
                        self.data.actuator_force[
                            mujoco.mj_name2id(
                                self.model,
                                mujoco.mjtObj.mjOBJ_ACTUATOR,
                                actuator_list[i],
                            )
                        ]
                        * force_scale,
                    ],
                    rgba=rgba_list,
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    label="",
                )
        else:
            for i in range(0, len(site_list)):
                self.add_marker(
                    pos=self.data.site(i).xpos,
                    mat=self.data.site(i).xmat,
                    size=[
                        arrow_radius,
                        arrow_radius,
                        self.data.actuator_force[
                            mujoco.mj_name2id(
                                self.model,
                                mujoco.mjtObj.mjOBJ_ACTUATOR,
                                actuator_list[i],
                            )
                        ]
                        * force_scale,
                    ],
                    rgba=rgba_list,
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    label=str(actuator_list[i])
                    + ":"
                    + str(
                        self.data.actuator_force[
                            mujoco.mj_name2id(
                                self.model,
                                mujoco.mjtObj.mjOBJ_ACTUATOR,
                                actuator_list[i],
                            )
                        ]
                    ),
                )

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
            "[J]oints",
            "On" if self._joints else "Off")
        add_overlay(
            topleft,
            "[I]nertia",
            "On" if self._inertias else "Off")
        add_overlay(
            topleft,
            "Center of [M]ass",
            "On" if self._com else "Off")
        add_overlay(
            topleft, "Shad[O]ws", "On" if self._shadows else "Off"
        )
        add_overlay(
            topleft,
            "T[r]ansparent",
            "On" if self._transparent else "Off")
        add_overlay(
            topleft,
            "[W]ireframe",
            "On" if self._wire_frame else "Off")
        add_overlay(
            topleft,
            "Con[V]ex Hull Rendering",
            "On" if self._convex_hull_rendering else "Off",
        )
        if self._paused is not None:
            if not self._paused:
                add_overlay(topleft, "Stop", "[Space]")
            else:
                add_overlay(topleft, "Start", "[Space]")
                add_overlay(
                    topleft,
                    "Advance simulation by one step",
                    "[right arrow]")
        add_overlay(topleft, "Toggle geomgroup visibility (0-5)",
                    ",".join(["On" if g else "Off" for g in self.vopt.geomgroup]))
        add_overlay(
            topleft,
            "Referenc[e] frames",
            mujoco.mjtFrame(self.vopt.frame).name)
        add_overlay(topleft, "[H]ide Menus", "")
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            add_overlay(topleft, "Cap[t]ure frame", "Saved as %s" % fname)
        else:
            add_overlay(topleft, "Cap[t]ure frame", "")

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

    def apply_perturbations(self):
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def read_pixels(self, camid=None):
        if self.render_mode == 'window':
            raise NotImplementedError(
                "Use 'render()' in 'window' mode.")

        if camid is not None:
            if camid == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camid

        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
            self.window)
        # update scene
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scn)
        # render
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)

        img = np.zeros(
            (glfw.get_framebuffer_size(
                self.window)[1], glfw.get_framebuffer_size(
                self.window)[0], 3), dtype=np.uint8)
        mujoco.mjr_readPixels(img, None, self.viewport, self.ctx)
        return np.flipud(img)

    def render(self):
        if self.render_mode == 'offscreen':
            raise NotImplementedError(
                "Use 'read_pixels()' for 'offscreen' mode.")
        if not self.is_alive:
            raise Exception(
                "GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window)
            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn)
                # marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # render
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)
                # overlay items
                for gridpos, [t1, t2] in self._overlay.items():
                    menu_positions = [mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      mujoco.mjtGridPos.mjGRID_BOTTOMLEFT]
                    if gridpos in menu_positions and self._hide_menus:
                        continue

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
                if glfw.window_should_close(self.window):
                    self.close()
                    break
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

        # apply perturbation (should this come before mj_step?)
        self.apply_perturbations()

    def close(self):
        self.is_alive = False
        glfw.terminate()
        self.ctx.free()
