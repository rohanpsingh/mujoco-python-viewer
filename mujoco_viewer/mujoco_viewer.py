import mujoco
import glfw
import numpy as np
import time
import pathlib
import yaml
from .callbacks import Callbacks
# import imgui
# from imgui.integrations.glfw import GlfwRenderer

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
        if hide_menus is True:
            self._hide_graph = True

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
        
        # IMGUI
        # imgui.create_context()
        # io = imgui.get_io()
        # self.impl = GlfwRenderer(self.window)
        
        # widgets_basic_f1_1= 0.0
        # changed, widgets_basic_f1_1 = imgui.slider_float(
        #         label="slider float",
        #         value=widgets_basic_f1_1,
        #         min_value=0.0,
        #         max_value=1.0,
        #         format="ratio = %.3f",
        #     )

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
        self.fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(self.fig)
        
        # Points for sampling of sensors... dictates smoothness of graph
        self._num_pnts = 100
        self._data_graph_line_names = []
        self._line_datas = []
        
        for n in range(0, len(self.model.sensor_adr) * 3):
            for i in range(0, 300):
                self.fig.linedata[n][2 * i] = float(-i)
        
        self.ctx = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        # Adjust placement and size of graph
        width, height = glfw.get_framebuffer_size(self.window)
        width_adjustment = width % 4
        self.graph_viewport = mujoco.MjrRect(
            int(3 * width / 4) + width_adjustment,
            0,
            int(width / 4),
            int(height / 4),
        )
        mujoco.mjr_figure(self.graph_viewport, self.fig, self.ctx)
        self.fig.flg_extend = 1
        self.fig.flg_symmetric = 0
        
        # Makes the graph to be in autorange
        self.axis_autorange()

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

    def set_grid_divisions(self, x_div: int, y_div: int, x_axis_time: float = 0.0, override=False):
        if override is False:
            assert x_axis_time >= self.model.opt.timestep * 50, "Set [x_axis_time] >= [self.model.opt.timestep * 50], inorder to get a suitable sampling rate"  
        self.fig.gridsize[0] = x_div + 1
        self.fig.gridsize[1] = y_div + 1
        if x_axis_time != 0.0:
            self._num_pnts = x_axis_time / self.model.opt.timestep
            print("self._num_pnts: ", self._num_pnts)
            if self._num_pnts > 300:
                self._num_pnts = 300
                new_x_axis_time = self.model.opt.timestep * self._num_pnts
                print(
                    f"Minimum x_axis_time is: {new_x_axis_time}"
                    + " reduce the x_axis_time"
                    f" OR Maximum time_step is: "
                    + f"{self.model.opt.timestep*self._num_pnts}"
                    + " increase the timestep"
                )
                # assert x_axis_time ==
            assert 1 <= self._num_pnts <= 300, (
                "num_pnts should be [10,300], it is currently:",
                f"{self._num_pnts}",
            )
            # self._num_pnts = num_pnts
            self._time_per_div = (self.model.opt.timestep * self._num_pnts) / (
                x_div
            )
            self.set_x_label(
                xname=f"time/div: {self._time_per_div}s"
                + f" total: {self.model.opt.timestep * self._num_pnts}"
            )
    
    def axis_autorange(self):
        """
        Call this function to auto-range the graph 
        """
        self.fig.range[0][0] = 1.0
        self.fig.range[0][1] = -1.0
        self.fig.range[1][0] = 1.0
        self.fig.range[1][1] = -1.0

    def set_graph_name(self, name: str):
        assert type(name) == str, "name is not a string"
        self.fig.title = name

    def show_graph_legend(self, show_legend: bool = True):
        if show_legend is True:
            for i in range(0, len(self._data_graph_line_names)):
                self.fig.linename[i] = self._data_graph_line_names[i]
            self.fig.flg_legend = True

    def set_x_label(self, xname: str):
        assert type(xname) == str, "xname is not a string"
        self.fig.xlabel = xname

    def add_graph_line(self, line_name, line_data=0.0):
        assert (
            type(line_name) == str
        ), f"Line_name is not a string: {type(line_name)}"
        if line_name in self._data_graph_line_names:
            print("line name already exists")
        else:
            self._data_graph_line_names.append(line_name)
            self._line_datas.append(line_data)

    def update_graph_line(self, line_name, line_data):
        if line_name in self._data_graph_line_names:
            idx = self._data_graph_line_names.index(line_name)
            self._line_datas[idx] = line_data
        else:
            raise NameError(
                "line name is not valid, add it to list before calling update"
            )

    def sensorupdate(self):
        pnt = int(mujoco.mju_min(self._num_pnts, self.fig.linepnt[0] + 1))
        
        for n in range(0, len(self._line_datas)):
            for i in range(pnt - 1, 0, -1):
                self.fig.linedata[n][2 * i + 1] = self.fig.linedata[n][
                    2 * i - 1
                ]
            self.fig.linepnt[n] = pnt
            self.fig.linedata[n][1] = self._line_datas[n]
            
    def update_graph_size(self, size_div_x=None, size_div_y=None):
        if size_div_x is None and size_div_y is None:
            width, height = glfw.get_framebuffer_size(self.window)
            width_adjustment = width % 3
            self.graph_viewport.left = int(2 * width / 3) + width_adjustment
            self.graph_viewport.width = int(width / 3)
            self.graph_viewport.height = int(height / 3)

        else:
            assert size_div_x is not None and size_div_y is None, ""
            width, height = glfw.get_framebuffer_size(self.window)
            width_adjustment = width % size_div_x
            self.graph_viewport.left = (
                int((size_div_x - 1) * width / size_div_x) + width_adjustment
            )
            self.graph_viewport.width = int(width / size_div_x)
            self.graph_viewport.height = int(height / size_div_x)

    def show_actuator_forces(
        self,
        f_render_list,
        rgba_list=[1, 0, 1, 1],
        force_scale=0.05,
        arrow_radius=0.03,
        show_force_labels=False,
    ) -> None:
        """f_render_list: [  ["jnt_name1","act_name_1","lable1"] ,
                             ["jnt_name2","act_name_2","lable2"] ]
        """
        if show_force_labels is False:
            for i in range(0, len(f_render_list)):
                self.add_marker(
                    pos=self.data.joint(f_render_list[i][0]).xanchor,
                    mat=self.rotation_matrix_from_vectors(
                        vec1=[0.0, 0.0, 1.0],
                        vec2=self.data.joint(f_render_list[i][0]).xaxis),
                    size=[
                        arrow_radius,
                        arrow_radius,
                        self.data.actuator(f_render_list[i][1]).force* force_scale,
                    ],
                    rgba=rgba_list,
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    label="",
                )
        else:
            for i in range(0, len(f_render_list)):
                self.add_marker(
                    pos=self.data.joint(f_render_list[i][0]).xanchor,
                    mat=self.rotation_matrix_from_vectors(
                        vec1=[0.0, 0.0, 1.0],
                        vec2=self.data.joint(f_render_list[i][0]).xaxis),
                    size=[
                        arrow_radius,
                        arrow_radius,
                        self.data.actuator(f_render_list[i][1]).force* force_scale,
                    ],
                    rgba=rgba_list,
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    label=f_render_list[i][2]
                    + ":"
                    + str(self.data.actuator(f_render_list[i][1]).force[0]),
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
            "[G]raph Viewer",
            "Off" if self._hide_graph else "On")
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
                
                # Handle graph and pausing interactions
                if (
                    not self._paused
                    and not self._hide_graph
                ):
                    self.sensorupdate()
                    self.update_graph_size()
                    mujoco.mjr_figure(
                        self.graph_viewport, self.fig, self.ctx
                    )
                elif self._hide_graph and self._paused:
                    self.update_graph_size()
                elif not self._hide_graph and self._paused:
                    mujoco.mjr_figure(
                        self.graph_viewport, self.fig, self.ctx
                    )
                elif self._hide_graph and not self._paused:
                    self.sensorupdate()
                    self.update_graph_size()

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
        # self.impl.shutdown()
        glfw.terminate()
        self.ctx.free()
        
    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        # vec1 = np.array(vec1,dtype=object)
        # vec2 = np.array(vec2,dtype=object)
        print(f"vec1: {vec1}")
        print(f"vec2: {vec2}")
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
