import mujoco
import mujoco_viewer
import math

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option gravity="0 0 -9.81" timestep="0.002" />
    <asset>
        <material name="blue" rgba="0 0 1 1" />
        <material name="green" rgba="0 1 0 1" />
        <material name="red" rgba="1 0 0 1" />
        <material name="white" rgba="1 1 1 1" />
    </asset>
    <worldbody>
        <geom type="plane" size="10 10 0.1" pos="0 0 -1.6" rgba=".3 .3 .3 1" />
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1" />

        <body name="link_1" pos="0 0 0" euler="90 0 0">
            <joint name="hinge_1" type="hinge" axis="1 0 0" />
            <geom name="g_link_1" type="cylinder" size=".1 .5" pos="0 0 .5" euler="0 0 0" material="red" mass="1" />
            <body name="link_2" pos=".2 0 1" euler="0 0 0">
                <joint name="hinge_2" type="hinge" axis="0 1 0" />
                <geom name="g_link_2" type="cylinder" size=".1 .5" pos="0 0 .5" euler="0 0 0" material="green" mass="1" />
                <body name="link_3" pos=".2 0 1" euler="0 0 0">
                    <joint name="hinge_3" type="slide" axis="1 0 0" />
                    <geom name="g_link_3" type="cylinder" size=".1 .5" pos="0.3 0 0" euler="0 90 0" material="blue" mass="1" />
                    <geom name="end_mass" type="sphere" size=".25" pos="1 0 0" euler="0 0 0" material="blue" mass="1" />
                </body>
            </body>
        </body>

    </worldbody>
    <actuator>
        <position name="pos_servo_1" joint="hinge_1" kp="100" />
        <position name="pos_servo_2" joint="hinge_2" kp="100" />
        <position name="pos_servo_3" joint="hinge_3" kp="100" />
    </actuator>
    <sensor>
        <jointpos name="pos_sensor_1" joint="hinge_1" />
        <jointpos name="pos_sensor_2" joint="hinge_2" />
        <jointpos name="pos_sensor_3" joint="hinge_3" />
    </sensor>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(MODEL_XML)
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)

f_render_list = [
    ["hinge_1", "pos_servo_1", "force_at_hinge_1"],
    ["hinge_2", "pos_servo_2", "force_at_hinge_2"],
    ["hinge_3", "pos_servo_3", "force_at_hinge_3"],
]

for _ in range(10000):
    # Render forces
    viewer.show_actuator_forces(
        f_render_list=f_render_list,
        rgba_list=[1, 0.5, 1, 0.5],
        force_scale=0.05,
        arrow_radius=0.05,
        show_force_labels=True,
    )

    # step and render
    mujoco.mj_step(model, data)
    viewer.render()
    if not viewer.is_alive:
        break

# close
viewer.close()
