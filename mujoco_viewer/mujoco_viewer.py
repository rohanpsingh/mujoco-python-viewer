#! /usr/bin/env python
""" `mujoco_viewer.py`

    @author:  Jack (Jianxiang) Xu
        Contacts    : projectbyjx@gmail.com
        Last edits  : July 27, 2022

    @description:
        This library will provide rendering pipeline (thread-safe)
"""
#===================================#
#  I M P O R T - L I B R A R I E S  #
#===================================#

# python libraries:
from ast import Lambda
from turtle import left
import numpy as np
import time
import imageio
import copy
import sys

from typing import Optional, Dict, Union, Any, Tuple
from enum import IntFlag, auto
from dataclasses import dataclass

# python 3rd party libraries:
import mujoco
import glfw
from threading import Lock

from icecream import ic


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #        ___           ___         ___          ___           ___           ___       # #
# #       /__/\         /__/\       /  /\        /  /\         /  /\         /  /\      # #
# #      |  |::\        \  \:\     /  /:/       /  /::\       /  /:/        /  /::\     # #
# #      |  |:|:\        \  \:\   /__/::\      /  /:/\:\     /  /:/        /  /:/\:\    # #
# #    __|__|:|\:\   ___  \  \:\  \__\/\:\    /  /:/  \:\   /  /:/  ___   /  /:/  \:\   # #
# #   /__/::::| \:\ /__/\  \__\:\    \  \:\  /__/:/ \__\:\ /__/:/  /  /\ /__/:/ \__\:\  # #
# #   \  \:\~~\__\/ \  \:\ /  /:/     \__\:\ \  \:\ /  /:/ \  \:\ /  /:/ \  \:\ /  /:/  # #
# #    \  \:\        \  \:\  /:/      /  /:/  \  \:\  /:/   \  \:\  /:/   \  \:\  /:/   # #
# #     \  \:\        \  \:\/:/      /__/:/    \  \:\/:/     \  \:\/:/     \  \:\/:/    # #
# #      \  \:\        \  \::/       \__\/      \  \::/       \  \::/       \  \::/     # #
# #       \__\/         \__\/                    \__\/         \__\/         \__\/      # #
# #                                                                                     # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#=======================#
#  D E F I N I T I O N  #
#=======================#
class _VIEWER_STATUS_MAP(IntFlag):
    # Status:
    IS_ALIVE                = (1<<0)
    # Visual:
    COM                     = (1<<2)
    JOINTS                  = (1<<3)
    SHADOWS                 = (1<<4)
    CONTACTS                = (1<<5)
    INERTIAS                = (1<<6)
    WIRE_FRAME              = (1<<7)
    TRANSPARENT             = (1<<8)
    RENDER_ON_SCREEN        = (1<<9)
    TOGGLE_COORD_FRAME      = (1<<10) 
    CONVEX_HULL_RENDERING   = (1<<11)
    HIDE_MENUS              = (1<<12)
    GROUP_0                 = (1<<13)
    GROUP_1                 = (1<<14)
    GROUP_2                 = (1<<15)
    GROUP_3                 = (1<<16)
    # MOD KEY:
    LEFT_SHIFT_PRESSED      = (1<<17)
    LEFT_CONTROL_PRESSED    = (1<<18)
    # Functional:
    TERMINATE               = (1<<1)
    CAPTURE_SCRNSHOT        = (1<<20)
    SAVE_SCRNSHOT_ON_FLY    = (1<<19)
    # TODO: to-be-implemented
    PAUSED                  = (1<<21) # TODO: to implement on Engine side (partially implemented here)
    STEP_BY_STEP            = (1<<22) # TODO: to implement on Engine side (unset bit unless engine resets it)
    RENDER_EVERY_FRAME      = (1<<23) # TODO: to be implemented
    # mux configs:
    DEFAULT                 = GROUP_0 | GROUP_1 | GROUP_2 | GROUP_3 | SHADOWS

@dataclass
class _CAMERA_DATA:
    frame_buffer = None
    mj_cam       = None
    mj_viewport  = None
    frame_stamp  = None

class MujocoViewer:
    #=========================#
    #  P L A C E H O L D E R  #
    #=========================#
    @dataclass
    class _MOUSE_DATA:
        button_left_pressed         = False
        button_right_pressed        = False
        left_double_click_pressed   = False
        right_double_click_pressed  = False
        last_left_click_time        = None
        last_right_click_time       = None
        last_mouse_click_x          = 0
        last_mouse_click_y          = 0
        dx                          = 0
        dy                          = 0
        last_x                      = 0
        last_y                      = 0
        y_offset_transient          = 0
        button_released             = False
    @dataclass
    class _VIEWER_CONFIG:
        title: str                                      = "DEFAULT"
        window_size: Optional[Tuple[int, int]]          = None
        image_output_format: str                        = "output/frame_%07d.png"
        # FPS: float                                      = 60.0 [NOT IMPLEMENTED]
        simulation_run_speed: float                     = 1.0    # [UNUSED]
        sensor_config                                   = dict()
        fixedcamid: int                                 = -1
        N_CAM: int                                      = 0
        _viewer_status: _VIEWER_STATUS_MAP              = _VIEWER_STATUS_MAP.DEFAULT
        
        def index_camera_unsafe(self):
            self.fixedcamid += 1
            if self.fixedcamid >= self.N_CAM:
                self.fixedcamid = -1
        
        def scale_simulation_run_speed_unsafe(self, factor):
            self.simulation_run_speed *= factor
            
        def update_viewer_status_unsafe(self, status, if_enable):
            if if_enable:
                self._viewer_status |= status
            else:
                self._viewer_status &= ~status
        
        def toggle_viewer_status_unsafe(self, status):
            self._viewer_status ^= status
        
        def set_viewer_status_unsafe(self, status):
            self._viewer_status |= status
        
        def clear_viewer_status_unsafe(self, status):
            self._viewer_status &= ~status

        def get_viewer_status_unsafe(self, status):
            return bool(self._viewer_status & status)
    @dataclass
    class _MJ_DATA:
        # cache:
        model        = None
        data         = None
        # mujoco placeholder:
        cam          = None
        vopt         = None
        scn          = None
        pert         = None
        ctx          = None
        camera_data: _CAMERA_DATA  = None
    @dataclass
    class _GUI_DATA:
        # glfw placeholder:
        glfw_window        = None
        frame_buffer_size  = None
        frame_window_size  = None
        frame_scale        = None
        viewer_overlay     = None
        viewer_markers     = None
        frame_rendering_time = 1
        frame_stamp          = 0
        
    #===============================#
    #  I N I T I A L I Z A T I O N  #
    #===============================#
    # internal:    
    _mouse_data_lock                    = Lock()
    _mouse_data: _MOUSE_DATA            = _MOUSE_DATA()
    
    _viewer_config_lock                 = Lock() 
    _viewer_config: _VIEWER_CONFIG      = _VIEWER_CONFIG()
    
    _mj_lock                            = Lock()
    _mj: _MJ_DATA                       = _MJ_DATA()
    
    _gui_lock                           = Lock()
    _gui_data: _GUI_DATA                = _GUI_DATA()
    
    _camera_data_lock  = Lock()
    _camera_data: Dict[str, _CAMERA_DATA] = {}
    
    def __init__(self,
        mj_model,
        mj_data,
        sensor_config                                   = None,
        title: str                                      = "unnamed",
        image_output_format: str                        = "output/frame_%07d.png",
        window_size: Optional[Tuple[int, int]]          = None,
        if_hide_menus: bool                             = True,
        if_render_on_screen: bool                       = True,
        if_save_scrnshot_on_fly: bool                   = True,
    ):
        # update config:
        if sensor_config:
            self._viewer_config.sensor_config = sensor_config
        self._viewer_config.title = title
        self._viewer_config.window_size = window_size
        self._viewer_config.update_viewer_status_unsafe(_VIEWER_STATUS_MAP.HIDE_MENUS, if_hide_menus)
        self._viewer_config.update_viewer_status_unsafe(_VIEWER_STATUS_MAP.RENDER_ON_SCREEN, if_render_on_screen)
        self._viewer_config.N_CAM = mj_model.ncam
        self._viewer_config.update_viewer_status_unsafe(_VIEWER_STATUS_MAP.SAVE_SCRNSHOT_ON_FLY, if_save_scrnshot_on_fly)
        self._viewer_config.image_output_format = image_output_format
        
        # cache mj objects:
        self._mj.model       = mj_model
        self._mj.data        = mj_data
        
        ## module initialization sequence:
        self._init_glfw_safe() # depends on config
        self._init_Mujoco_safe() # depends on glfw

    def _init_Mujoco_safe(self):
        # get viewport
        with self._viewer_config_lock:
            ww, wh = self._viewer_config.window_size
        
        # create options, camera, scene, context
        with self._mj_lock:
            self._mj.vopt    = mujoco.MjvOption()
            self._mj.scn     = mujoco.MjvScene(self._mj.model, maxgeom=10000)
            self._mj.pert    = mujoco.MjvPerturb()
            self._mj.ctx     = mujoco.MjrContext(self._mj.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            # camera buffer:
            self._mj.camera_data              = _CAMERA_DATA()
            self._mj.camera_data.mj_cam       = mujoco.MjvCamera()
            self._mj.camera_data.mj_viewport  = mujoco.MjrRect(0, 0, ww, wh)
            self._mj.camera_data.frame_buffer = np.zeros((wh, ww, 3), dtype=np.uint8)
            self._mj.camera_data.frame_stamp  = self._gui_data.frame_stamp
        
        # initialize overlay, markers
        with self._gui_lock:
            self._gui_data.viewer_overlay = {}
            self._gui_data.viewer_markers = []
            
        # create camera sensor buffers:
        for camera in self._viewer_config.sensor_config:
            with self._viewer_config_lock:
                w = self._viewer_config.sensor_config[camera]["width"]
                h = self._viewer_config.sensor_config[camera]["height"]
            # Create Buffer:
            with self._camera_data_lock:
                self._camera_data[camera].mj_viewport   = mujoco.MjrRect(0, 0, w, h)
                self._camera_data[camera].frame_buffer  = np.zeros((h, w, 3), dtype=np.uint8)
                self._camera_data[camera].frame_stamp   = self._gui_data.frame_stamp
                self._camera_data[camera].mj_cam        = mujoco.MjvCamera()
                self._camera_data[camera].mj_cam.type          = mujoco.mjtCamera.mjCAMERA_FIXED
                self._camera_data[camera].mj_cam.fixedcamid    = self._sensor_cameras[camera]["id"]
        
    def _init_glfw_safe(self):
        # - fetch config:
        with self._viewer_config_lock:
            window_size = self._viewer_config.window_size
            title       = self._viewer_config.title
            if_on_scrn  = self._viewer_config.get_viewer_status_unsafe(_VIEWER_STATUS_MAP.RENDER_ON_SCREEN)
        
        # - init:
        glfw.init()
        
        # - Create Window:
        if not window_size: # auto-window
            window_size = glfw.get_video_mode(glfw.get_primary_monitor()).size
            with self._viewer_config_lock:
                self._viewer_config.window_size = window_size
        ww, wh = window_size
        
        glfw_window = glfw.create_window(ww, wh, title, None, None)
        
        glfw.window_hint(glfw.VISIBLE, if_on_scrn)
        glfw.make_context_current(glfw_window)
        glfw.swap_interval(1)

        frame_buffer_size = glfw.get_framebuffer_size(glfw_window)
        frame_window_size = glfw.get_window_size(glfw_window)
        frame_scale = frame_buffer_size[0] * 1.0 / frame_window_size[0]
        
        # - cache:
        with self._gui_lock:
            self._gui_data.glfw_window = glfw_window
            self._gui_data.frame_buffer_size = frame_buffer_size
            self._gui_data.frame_window_size = frame_window_size
            self._gui_data.frame_scale = frame_scale
            
        # - initialize interactive on-screen modules:
        if if_on_scrn:
            # - set callbacks
            glfw.set_cursor_pos_callback        (glfw_window, self._cursor_pos_callback       )
            glfw.set_mouse_button_callback      (glfw_window, self._mouse_button_callback     )
            glfw.set_scroll_callback            (glfw_window, self._scroll_callback           )
            glfw.set_key_callback               (glfw_window, self._key_callback              )
            glfw.set_framebuffer_size_callback  (glfw_window, self._framebuffer_size_callback )
        
        with self._viewer_config_lock:
            self._viewer_config.set_viewer_status_unsafe(_VIEWER_STATUS_MAP.IS_ALIVE)

    #===================#
    #  C A L L B A C K  #
    #===================# 
    def _framebuffer_size_callback(self, window, width, height):
        frame_buffer_size = glfw.get_framebuffer_size(window)
        frame_window_size = glfw.get_window_size(window)
        frame_scale = frame_buffer_size[0] * 1.0 / frame_window_size[0]
        # - cache:
        with self._gui_lock:
            self._gui_data.frame_buffer_size = frame_buffer_size
            self._gui_data.frame_window_size = frame_window_size
            self._gui_data.frame_scale = frame_scale
        
    # === CALLBACK MAP  === #
    _key_stroke_callback_map = {
        glfw.KEY_TAB        : lambda config: config.index_camera_unsafe(),
        glfw.KEY_SPACE      : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.PAUSED),
        glfw.KEY_RIGHT      : lambda config: config.set_viewer_status_unsafe    (_VIEWER_STATUS_MAP.PAUSED | _VIEWER_STATUS_MAP.STEP_BY_STEP),
        glfw.KEY_S          : lambda config: config.scale_simulation_run_speed_unsafe(0.5), # slow down
        glfw.KEY_F          : lambda config: config.scale_simulation_run_speed_unsafe(2.0), # faster
        glfw.KEY_D          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.RENDER_EVERY_FRAME),
        glfw.KEY_P          : lambda config: config.set_viewer_status_unsafe    (_VIEWER_STATUS_MAP.CAPTURE_SCRNSHOT),
        glfw.KEY_C          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.CONTACTS),
        glfw.KEY_J          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.JOINTS),
        glfw.KEY_R          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.TOGGLE_COORD_FRAME),
        glfw.KEY_H          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.HIDE_MENUS),
        glfw.KEY_T          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.TRANSPARENT),
        glfw.KEY_I          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.INERTIAS),
        glfw.KEY_M          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.COM),
        glfw.KEY_O          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.SHADOWS),
        glfw.KEY_V          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.CONVEX_HULL_RENDERING),
        glfw.KEY_W          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.WIRE_FRAME),
        glfw.KEY_ESCAPE     : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.TERMINATE),
        glfw.KEY_LEFT_SHIFT : lambda config: config.clear_viewer_status_unsafe  (_VIEWER_STATUS_MAP.LEFT_SHIFT_PRESSED), # release
        glfw.KEY_LEFT_CONTROL: lambda config: config.clear_viewer_status_unsafe  (_VIEWER_STATUS_MAP.LEFT_CONTROL_PRESSED), # release
        # Number keys
        glfw.KEY_0          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.GROUP_0),
        glfw.KEY_1          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.GROUP_1),
        glfw.KEY_2          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.GROUP_2),
        glfw.KEY_3          : lambda config: config.toggle_viewer_status_unsafe (_VIEWER_STATUS_MAP.GROUP_3),
    }
    def _key_callback(self, window, key, scancode, action, mods):    
        
        if action == glfw.RELEASE:
            # register key upon release
            if key in self._key_stroke_callback_map:
                with self._viewer_config_lock:
                    self._key_stroke_callback_map[key](self._viewer_config)
            else:
                pass # unregistered key press
        else:
            if key == glfw.KEY_LEFT_SHIFT:
                with self._viewer_config_lock:
                    self._viewer_config.set_viewer_status_unsafe(_VIEWER_STATUS_MAP.LEFT_SHIFT_PRESSED)
            elif key == glfw.KEY_LEFT_CONTROL:
                with self._viewer_config_lock:
                    self._viewer_config.set_viewer_status_unsafe(_VIEWER_STATUS_MAP.LEFT_CONTROL_PRESSED)
        return

    def _cursor_pos_callback(self, window, mouse_x, mouse_y):
        with self._mouse_data_lock:
            self._mouse_data.dx = mouse_x - self._mouse_data.last_x
            self._mouse_data.dy = mouse_y - self._mouse_data.last_y
            self._mouse_data.last_x = mouse_x
            self._mouse_data.last_y = mouse_y

    def _mouse_button_callback(self, window, button, act, mods):
        # Local:
        left_double_click_pressed = False
        right_double_click_pressed = False
        last_left_click_time = None
        last_right_click_time = None
        
        # read cache:
        with self._mouse_data_lock:
            last_left_click_time = self._mouse_data.last_left_click_time
            last_right_click_time = self._mouse_data.last_right_click_time
        
        # process:
        time_now = glfw.get_time()
        button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS
        
        if button_left_pressed:
            if last_left_click_time:
                delta_t = time_now - last_left_click_time
                if 0.01 < delta_t < 0.3:
                    left_double_click_pressed = True            
            last_left_click_time = time_now
        if button_right_pressed:
            if last_right_click_time:
                delta_t = time_now - last_right_click_time
                if 0.01 < delta_t < 0.3:
                    right_double_click_pressed = True            
            last_right_click_time = time_now
        
        x, y = glfw.get_cursor_pos(window)
        # cache:
        with self._mouse_data_lock:
            self._mouse_data.last_mouse_click_x = x
            self._mouse_data.last_mouse_click_y = y
            self._mouse_data.button_left_pressed = button_left_pressed
            self._mouse_data.button_right_pressed = button_right_pressed
            self._mouse_data.left_double_click_pressed = left_double_click_pressed
            self._mouse_data.right_double_click_pressed = right_double_click_pressed
            self._mouse_data.last_left_click_time = last_left_click_time
            self._mouse_data.last_right_click_time = last_right_click_time
            self._mouse_data.button_released = (act == glfw.RELEASE)
            
    def _scroll_callback(self, window, x_offset, y_offset_transient):
        with self._mouse_data_lock:
            self._mouse_data.y_offset_transient = y_offset_transient
    
    #====================================#
    #  P R I V A T E    F U N C T I O N  #
    #====================================#
    def _on_terminate_safe(self):
        print("Terminating mujoco_viewer!")
        glfw.set_window_should_close(self._gui_data.glfw_window, True)
        glfw.terminate()
        with self._mj_lock:
            self._mj.ctx.free()
        with self._viewer_config_lock:
            self._viewer_config.clear_viewer_status_unsafe(_VIEWER_STATUS_MAP.IS_ALIVE)
        print("mujoco_viewer has been Terminated!")
        sys.exit()

    def _process_viewer_config(self):
        # - Fetch Inputs:
        with self._viewer_config_lock:
            current_viewer_status = self._viewer_config._viewer_status
            current_fixed_id = self._viewer_config.fixedcamid
        with self._gui_lock:
            frame_buffer_size = self._gui_data.frame_buffer_size
        
        # - Update MJ:
        with self._mj_lock:
            ### change cam id view:
            self._mj.camera_data.mj_cam.fixedcamid = current_fixed_id
            if self._mj.camera_data.mj_cam.fixedcamid == -1:
                self._mj.camera_data.mj_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self._mj.camera_data.mj_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            ### coord frame toggle:
            if bool(current_viewer_status & _VIEWER_STATUS_MAP.TOGGLE_COORD_FRAME):
                self._mj.vopt.frame = 1 - self._mj.vopt.frame
                current_viewer_status ^= ~(_VIEWER_STATUS_MAP.TOGGLE_COORD_FRAME) # clear status
            ### transparency:
            if bool(current_viewer_status & _VIEWER_STATUS_MAP.TRANSPARENT):
                self._mj.model.geom_rgba[:, 3] = 0.5
            else:
                self._mj.model.geom_rgba[:, 3] = 1.0
            ### visual features:
            self._mj.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = bool(current_viewer_status & _VIEWER_STATUS_MAP.CONTACTS)
            self._mj.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = bool(current_viewer_status & _VIEWER_STATUS_MAP.CONTACTS)
            self._mj.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]        = bool(current_viewer_status & _VIEWER_STATUS_MAP.JOINTS)
            self._mj.vopt.flags[mujoco.mjtVisFlag.mjVIS_COM]          = bool(current_viewer_status & _VIEWER_STATUS_MAP.INERTIAS)
            self._mj.vopt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA]      = bool(current_viewer_status & _VIEWER_STATUS_MAP.COM)
            self._mj.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL]   = bool(current_viewer_status & _VIEWER_STATUS_MAP.CONVEX_HULL_RENDERING)
            ### OpenGL rendering effects.
            self._mj.scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = bool(current_viewer_status & _VIEWER_STATUS_MAP.WIRE_FRAME)
            self._mj.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]    = bool(current_viewer_status & _VIEWER_STATUS_MAP.SHADOWS)
            ### Geom Group:
            self._mj.vopt.geomgroup[0] = bool(current_viewer_status & _VIEWER_STATUS_MAP.GROUP_0)
            self._mj.vopt.geomgroup[1] = bool(current_viewer_status & _VIEWER_STATUS_MAP.GROUP_1)
            self._mj.vopt.geomgroup[2] = bool(current_viewer_status & _VIEWER_STATUS_MAP.GROUP_2)
            self._mj.vopt.geomgroup[3] = bool(current_viewer_status & _VIEWER_STATUS_MAP.GROUP_3)
            ### viewport dimension:
            self._mj.camera_data.mj_viewport.width, self._mj.camera_data.mj_viewport.height = frame_buffer_size
        
    def _process_mouse_interactions(self):
        """ Process mouse interactions:
            - selection
            - dragging interaction
            - camera angles and zoom
            
            TODO: double check the perturbation
        """
        # - local cache safely:
        with self._mouse_data_lock:
            current_mouse_data = copy.deepcopy(self._mouse_data)
        with self._viewer_config_lock:
            current_viewer_status = copy.deepcopy(self._viewer_config._viewer_status)
        with self._gui_lock:
            width, height = self._gui_data.frame_buffer_size
            scale = self._gui_data.frame_scale
            
        ### Selection ###
        # - mouse interaction:
        if_mod_control = bool(current_viewer_status & _VIEWER_STATUS_MAP.LEFT_CONTROL_PRESSED)
                
        # - perturbation:
        new_perturb = 0
        if if_mod_control and self._mj.pert.select > 0:
            # right: translate, left: rotate
            if current_mouse_data.button_right_pressed:
                new_perturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if current_mouse_data.button_left_pressed:
                new_perturb = mujoco.mjtPertBit.mjPERT_ROTATE
            # perturbation: reset reference
            if new_perturb and not self._mj.pert.active:
                mujoco.mjv_initPerturb(self._mj.model, self._mj.data, self._mj.scn, self._mj.pert)
        self._mj.pert.active = new_perturb
        
        # - interaction:
        selection_mode = 0
        if current_mouse_data.left_double_click_pressed:
            selection_mode = 1
        elif current_mouse_data.right_double_click_pressed:
            selection_mode = 3 if if_mod_control else 2
        
        if selection_mode:
            with self._mj_lock:
                vp_w, vp_h = self._mj.camera_data.mj_viewport.width, self._mj.camera_data.mj_viewport.height
            # find geom and 3D click point, get corresponding body
            aspect_ratio = vp_w / vp_h
            rel_x = int(scale * (current_mouse_data.last_mouse_click_x)) / vp_w
            rel_y = (vp_h - current_mouse_data.last_mouse_click_y) / vp_h
            selected_pnt = np.zeros((3, 1), dtype=np.float64)
            selected_geom = np.zeros((1, 1), dtype=np.int32)
            selected_skin = np.zeros((1, 1), dtype=np.int32)
            # apply selection:
            with self._mj_lock:
                selected_body = mujoco.mjv_select(
                    self._mj.model,
                    self._mj.data,
                    self._mj.vopt,
                    aspect_ratio,
                    rel_x,
                    rel_y,
                    self._mj.scn,
                    selected_pnt,
                    selected_geom,
                    selected_skin
                )

                # set lookat point, start tracking is requested
                if current_mouse_data.right_double_click_pressed:
                    # set cam lookat
                    if selected_body >= 0:
                        self._mj.camera_data.mj_cam.lookat = selected_pnt.flatten()
                    # switch to tracking camera if dynamic body clicked
                    if if_mod_control and selected_body > 0:
                        self._mj.camera_data.mj_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        self._mj.camera_data.mj_cam.trackbodyid = selected_body
                        self._mj.camera_data.mj_cam.fixedcamid = -1
                # set body selection
                else:
                    if selected_body >= 0:
                        # record selection
                        self._mj.pert.select = selected_body
                        self._mj.pert.skinselect = selected_skin
                        # compute localpos
                        vec = selected_pnt.flatten() - self._mj.data.xpos[selected_body]
                        mat = self._mj.data.xmat[selected_body].reshape(3, 3)
                        self._mj.pert.localpos = mat.dot(vec)
                    else:
                        self._mj.pert.select = 0
                        self._mj.pert.skinselect = -1
                # stop perturbation on select
                self._mj.pert.active = 0
            
        if current_mouse_data.button_released:
            # 3D release
            with self._mj_lock:
                self._mj.pert.active = 0

        ### Camera ###
        if current_mouse_data.button_left_pressed or current_mouse_data.button_right_pressed:
            # - process mouse interaction:
            if_mod = bool(current_viewer_status & _VIEWER_STATUS_MAP.LEFT_SHIFT_PRESSED)
            if current_mouse_data.button_left_pressed:
                mouse_action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if if_mod else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            elif current_mouse_data.button_right_pressed:
                mouse_action = mujoco.mjtMouse.mjMOUSE_MOVE_H if if_mod else mujoco.mjtMouse.mjMOUSE_MOVE_V
            
            dx = current_mouse_data.dx
            dy = current_mouse_data.dy
            dx_pix = int(scale * dx)
            dy_pix = int(scale * dy)
            
            # - update:
            with self._mj_lock:
                if self._mj.pert.active:
                    mujoco.mjv_movePerturb(
                        self._mj.model, self._mj.data,
                        mouse_action,
                        dx_pix / height,
                        dy_pix / height,
                        self._mj.scn, self._mj.pert)
                else:
                    mujoco.mjv_moveCamera(
                        self._mj.model,
                        mouse_action,
                        dx_pix / height,
                        dy_pix / height,
                        self._mj.scn, self._mj.camera_data.mj_cam)
        # - zoom:
        with self._mj_lock:
            mujoco.mjv_moveCamera(
                self._mj.model, 
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0, -0.05 * current_mouse_data.y_offset_transient, 
                self._mj.scn, self._mj.camera_data.mj_cam)
        
        with self._mouse_data_lock:
            self._mouse_data.y_offset_transient = 0

    @staticmethod
    def _add_marker_to_scene(marker, scn):
        if scn.ngeom >= scn.maxgeom:
            raise RuntimeError('Ran out of geoms. maxgeom: %d' % scn.maxgeom)

        g = scn.geoms[scn.ngeom]
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
        scn.ngeom += 1
        return

    def _create_overlay(self):
        with self._viewer_config_lock:
            v_status = copy.deepcopy(self._viewer_config._viewer_status)
            run_spd = self._viewer_config.simulation_run_speed
            N_cam = self._viewer_config.N_CAM
            image_output_format = self._viewer_config.image_output_format
        with self._mj_lock:
            camid = self._mj.camera_data.mj_cam.fixedcamid
            mj_timestep = self._mj.model.opt.timestep
            mj_data_time = self._mj.data.time
            mj_data_solviter = self._mj.data.solver_iter
        with self._gui_lock:
            frame_stamp = self._gui_data.frame_stamp
            frame_rendering_time = self._gui_data.frame_rendering_time
            
        if_render_every_frame    = bool(v_status & _VIEWER_STATUS_MAP.RENDER_EVERY_FRAME)
        if_contacts              = bool(v_status & _VIEWER_STATUS_MAP.CONTACTS)
        if_joints                = bool(v_status & _VIEWER_STATUS_MAP.JOINTS)
        if_inertias              = bool(v_status & _VIEWER_STATUS_MAP.INERTIAS)
        if_com                   = bool(v_status & _VIEWER_STATUS_MAP.COM)
        if_shadows               = bool(v_status & _VIEWER_STATUS_MAP.SHADOWS)
        if_transparent           = bool(v_status & _VIEWER_STATUS_MAP.TRANSPARENT)
        if_wire_frame            = bool(v_status & _VIEWER_STATUS_MAP.WIRE_FRAME)
        if_convex_hull_rendering = bool(v_status & _VIEWER_STATUS_MAP.CONVEX_HULL_RENDERING)
        if_paused                = bool(v_status & _VIEWER_STATUS_MAP.PAUSED)
        if_toggle_coord          = bool(v_status & _VIEWER_STATUS_MAP.TOGGLE_COORD_FRAME)
    
        _TOP_LEFT = mujoco.mjtGridPos.mjGRID_TOPLEFT
        _BTM_LEFT = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        
        group_avail = "{}{}{}{}".format(
            "[0]" if (v_status & _VIEWER_STATUS_MAP.GROUP_0) else "",
            "[1]" if (v_status & _VIEWER_STATUS_MAP.GROUP_1) else "",
            "[2]" if (v_status & _VIEWER_STATUS_MAP.GROUP_2) else "",
            "[3]" if (v_status & _VIEWER_STATUS_MAP.GROUP_3) else "",
        )
        overlay = {
            _TOP_LEFT: ["", ""],
            _BTM_LEFT: ["", ""],
        }

        _overlay_layout = {
            _TOP_LEFT: {
                "[S]lower, [F]aster"                                : "Run speed = %.3f x real time" % run_spd,
                "Ren[d]er every frame"                              : "On" if if_render_every_frame else "Off",
                "Switch camera (#cams = %d)" %(N_cam + 1)           : "[Tab] (camera ID = %d)" % camid,
                "[C]ontact forces"                                  : "On" if if_contacts else "Off",
                "[J]oints"                                          : "On" if if_joints else "Off",
                "[I]nertia"                                         : "On" if if_inertias else "Off",
                "Center of [M]ass"                                  : "On" if if_com else "Off",
                "Shad[O]ws"                                         : "On" if if_shadows else "Off",
                "[T]ransparent"                                     : "On" if if_transparent else "Off",
                "[W]ireframe"                                       : "On" if if_wire_frame else "Off",
                "Con[V]ex Hull Rendering"                           : "On" if if_convex_hull_rendering else "Off",
                "Stop" if if_paused else "Start"                    : "[Space]",
                "Step Through"                                      : "->[right arrow]" if if_paused else "[Space] to stop first",
                "[R]eference frames"                                : "On" if if_toggle_coord else "Off",
                "[H]ide Menus"                                      : "",
                "Ca[p]ture frame"                                   : ("path: " + image_output_format) % frame_stamp,
                "Toggle geomgroup visibility"                       : group_avail,
            },
            _BTM_LEFT: {
                "FPS"                 : "%d" % (1/frame_rendering_time),
                "Solver iterations"   : str(mj_data_solviter + 1),
                "Step"                : str(round(mj_data_time / mj_timestep)),
                "timestep"            : "%.5f" % mj_timestep,
            }
        }
        for grid_loc in _overlay_layout:
            for feature, description in _overlay_layout[grid_loc].items():
                overlay[grid_loc][0] += feature + "\n"
                overlay[grid_loc][1] += description + "\n"
        
        # override overlays:
        with self._gui_lock:
            self._gui_data.viewer_overlay = overlay

    #==================================#
    #  P U B L I C    F U N C T I O N  #
    #==================================#
    def signal_termination_safe(self, if_immediate:bool=False):
        """ Signal the viewer to self-destroy upon next update/rendering (Or immediately if True)
        """
        with self._viewer_config_lock:
            self._viewer_config.set_viewer_status_unsafe(_VIEWER_STATUS_MAP.TERMINATE)
        if if_immediate:
            self._on_terminate_safe()

    def add_marker_safe(self, **marker_params):
        with self._gui_lock:
            self._gui_data.viewer_markers.append(marker_params)

    def apply_perturbations_safe(self):
        """ Apply perturbation to mujoco: pose + force
        """
        with self._mj_lock:
            self._mj.data.xfrc_applied = np.zeros_like(self._mj.data.xfrc_applied)
            mujoco.mjv_applyPerturbPose(self._mj.model, self._mj.data, self._mj.pert, 0)
            mujoco.mjv_applyPerturbForce(self._mj.model, self._mj.data, self._mj.pert)

    def process_safe(self):
        """ Process: safely process configs from the physical / software modifications
        """
        self._process_viewer_config()
        self._process_mouse_interactions()
        self._create_overlay()    
        return
    
    def render_safe(self):
        """ Rendering: assume it has been udpated with `update_safe`
        """
        render_start = time.time()
        
        with self._viewer_config_lock:
            is_on_scrn = self._viewer_config.get_viewer_status_unsafe(_VIEWER_STATUS_MAP.RENDER_ON_SCREEN)
            is_alive = self._viewer_config.get_viewer_status_unsafe(_VIEWER_STATUS_MAP.IS_ALIVE)
            is_paused = self._viewer_config.get_viewer_status_unsafe(_VIEWER_STATUS_MAP.PAUSED)
            if_capture_req = self._viewer_config.get_viewer_status_unsafe(_VIEWER_STATUS_MAP.CAPTURE_SCRNSHOT)
            if_hide_menu = self._viewer_config.get_viewer_status_unsafe(_VIEWER_STATUS_MAP.HIDE_MENUS)
            if_terminate = self._viewer_config.get_viewer_status_unsafe(_VIEWER_STATUS_MAP.TERMINATE)

        # signal exit on window close:
        if_window_closed = False
        with self._gui_lock:
            if_window_closed = (glfw.window_should_close(self._gui_data.glfw_window))
            current_frame_stamp = self._gui_data.frame_stamp
            
        # process to determine if we shall exit():
        if if_terminate or if_window_closed:
            self.signal_termination_safe(if_immediate=True)
            return

        if not is_alive:
            self.signal_termination_safe(if_immediate=True)
            raise Exception("The GLFW window does not exist! Program is dead :(  ---> Terminating ...")
            
        with self._gui_lock:
            width, height = self._gui_data.frame_buffer_size
            scale = self._gui_data.frame_scale
        
        # - init:
        img_buffer = np.zeros((height, width, 3), dtype=np.uint8)

        # - update scene
        render_start = time.time()
        with self._mj_lock:
            mj_timestep = self._mj.model.opt.timestep
            mj_data_time = self._mj.data.time
            
            mujoco.mjv_updateScene(
                self._mj.model,
                self._mj.data,
                self._mj.vopt,
                self._mj.pert,
                self._mj.camera_data.mj_cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self._mj.scn
            )
            if is_on_scrn:
                # marker items
                with self._gui_lock:
                    for marker in self._gui_data.viewer_markers:
                        self._add_marker_to_scene(marker, self._mj.scn)
            
            # render
            mujoco.mjr_render(self._mj.camera_data.mj_viewport, self._mj.scn, self._mj.ctx)
            
            # read images:
            if if_capture_req:
                mujoco.mjr_readPixels(img_buffer, None, self._mj.camera_data.mj_viewport, self._mj.ctx)
                        
            # on-scrn render:
            if is_on_scrn:
                # overlay items
                with self._gui_lock:
                    for gridpos, [t1, t2] in self._gui_data.viewer_overlay.items():
                        # skip rendering for these locations if hidden:
                        if gridpos in [mujoco.mjtGridPos.mjGRID_TOPLEFT, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT] \
                            and if_hide_menu:
                            continue
                        # render overlay
                        mujoco.mjr_overlay(
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            gridpos, self._mj.camera_data.mj_viewport,
                            t1, t2, self._mj.ctx
                        )
                with self._gui_lock:
                    glfw.swap_buffers(self._gui_data.glfw_window)
                glfw.poll_events()
            else:
                print("TODO: Have not tested off creen mode")
            
        # clear gui data
        with self._gui_lock:
            self._gui_data.viewer_overlay.clear()
            self._gui_data.viewer_markers.clear()
            self._gui_data.frame_rendering_time = 0.9 * self._gui_data.frame_rendering_time + 0.1 * (time.time() - render_start)
            self._gui_data.frame_stamp = int(round(mj_data_time / mj_timestep))

        # apply perturbation (should this come before mj_step?)
        self.apply_perturbations_safe()
        
        if if_capture_req:
            # cache buffer:
            with self._mj_lock:
                # img_buffer_flipped = np.flipud(img_buffer)
                self._mj.camera_data.frame_buffer = img_buffer
                self._mj.camera_data.frame_stamp  = current_frame_stamp
            # reset capture flag:
            with self._viewer_config_lock:
                if_capture_req = self._viewer_config.clear_viewer_status_unsafe(_VIEWER_STATUS_MAP.CAPTURE_SCRNSHOT)
            
            # output:
            self.save_last_available_captured_scrnshot_safe()
    
        return
    
    def render_sensor_cameras_safe(self):
        for camera_name, camera in self._camera_data.items():
            # update scene
            with self._mj_lock:
                # change cam to the camera sensors 
                mujoco.mjv_updateScene(
                    self._mj.model,
                    self._mj.data,
                    self._mj.vopt,
                    self._mj.pert,
                    camera.mj_cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self._mj.scn)
            
            # render off-screen
            with self._camera_data_lock:
                mujoco.mjr_render(camera.mj_viewport, self._mj.scn, self._mj.ctx)

        for camera_name, camera in self._camera_data.items():
            with self._camera_data_lock:
                with self._mj_lock:
                    mujoco.mjr_readPixels(camera.frame_buffer, None, camera.mj_viewport, self._mj.ctx)
                    camera.frame_stamp = self._gui_data.frame_stamp

    def acquire_sensor_camera_frames_safe(self, write_to=None):
        camera_buffers = {"frame_buffer":{}, "frame_stamp":{}}
        with self._camera_data_lock:
            for camera_name, camera in self._camera_data_lock.items():
                camera_buffers["frame_buffer"][camera_name] = copy.deepcopy(camera.frame_buffer)
                camera_buffers["frame_stamp"][camera_name] = copy.deepcopy(camera.frame_stamp)
                if write_to:
                    imageio.imwrite(
                        "{}/{}.png".format(write_to, camera_name.replace("\\", "_")), 
                        camera_buffers["frame_buffer"][camera_name]
                    )
        return camera_buffers

    def save_last_available_captured_scrnshot_safe(self):
        with self._viewer_config_lock:
            if_new_capture_in_pending = self._viewer_config.get_viewer_status_unsafe(_VIEWER_STATUS_MAP.CAPTURE_SCRNSHOT)
            image_output_format = self._viewer_config.image_output_format
        
        if not if_new_capture_in_pending:
            with self._mj_lock:
                img_buffer = self._mj.camera_data.frame_buffer
                frame_stamp = self._mj.camera_data.frame_stamp
                if img_buffer is not None and frame_stamp > 0:
                    img_buffer_corrected = np.flipud(img_buffer)
                    # save image:
                    imageio.imwrite(image_output_format % frame_stamp, img_buffer_corrected)
                self._mj.camera_data.frame_stamp = -1 # prevent to capture again
        
    def is_key_registered_to_pause_program_safe(self):
        with self._viewer_config_lock:
            status = self._viewer_config.get_viewer_status_unsafe(_VIEWER_STATUS_MAP.PAUSED)
        return status

    def is_key_registered_to_step_to_next_safe(self):
        with self._viewer_config_lock:
            status = self._viewer_config.get_viewer_status_unsafe(_VIEWER_STATUS_MAP.STEP_BY_STEP)
        return status
    
    def reset_key_registered_to_step_to_next_safe(self):
        with self._viewer_config_lock:
            self._viewer_config.clear_viewer_status_unsafe(_VIEWER_STATUS_MAP.STEP_BY_STEP)
    