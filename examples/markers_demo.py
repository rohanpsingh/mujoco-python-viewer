import mujoco
import mujoco_viewer
import numpy as np

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
  <asset>
    <texture name="body" type="cube" builtin="flat" mark="cross" width="127" height="1278"
             rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
    <material name="body" texture="body" texuniform="true" rgba="0.8 0.6 .4 1"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>
    <worldbody>
        <geom name="floor" size="0 0 .05" type="plane" material="grid" condim="3"/>
        <body name="box" pos="0 0 1">
            <freejoint name="root"/>
            <geom size="0.15 0.15 0.15" type="box" material="body" />
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(MODEL_XML)
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

while viewer.is_alive:
    # sim step
    mujoco.mj_step(model, data)

    # draw origin
    x_dir = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    y_dir = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    z_dir = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    viewer.add_marker(
        pos=[
            0, 0, 0], size=[
            0.05, 0.05, 0.05], rgba=[
                1, 1, 1, 1], type=mujoco.mjtGeom.mjGEOM_SPHERE, label="origin")
    viewer.add_marker(
        pos=[
            0, 0, 0], mat=x_dir, size=[
            0.01, 0.01, 2], rgba=[
                1, 0, 0, 0.2], type=mujoco.mjtGeom.mjGEOM_ARROW, label="")
    viewer.add_marker(
        pos=[
            0, 0, 0], mat=y_dir, size=[
            0.01, 0.01, 2], rgba=[
                0, 1, 0, 0.2], type=mujoco.mjtGeom.mjGEOM_ARROW, label="")
    viewer.add_marker(
        pos=[
            0, 0, 0], mat=z_dir, size=[
            0.01, 0.01, 2], rgba=[
                0, 0, 1, 0.2], type=mujoco.mjtGeom.mjGEOM_ARROW, label="")

    # render
    viewer.render()

# close
viewer.close()
