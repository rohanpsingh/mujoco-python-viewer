import mujoco
import mujoco_viewer

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option gravity="0 0 -9.81" timestep="0.002" />
    <default>
        <site type="cylinder" size="0.01 0.03" zaxis="1 0 0" rgba="0.1 0.1 1 .1" />
    </default>

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
            <site name="site_1" />
            <geom name="g_link_1" type="cylinder" size=".1 .5" pos="0 0 .5" euler="0 0 0" material="red" mass="1" />
            <body name="link_2" pos=".2 0 1" euler="0 0 0">
                <joint name="hinge_2" type="hinge" axis="1 0 0" />
                <site name="site_2" />
                <geom name="g_link_2" type="cylinder" size=".1 .5" pos="0 0 .5" euler="0 0 0" material="green" mass="1" />
                <body name="link_3" pos=".2 0 1" euler="0 0 0">
                    <joint name="hinge_3" type="hinge" axis="1 0 0" />
                    <site name="site_3" />
                    <geom name="g_link_3" type="cylinder" size=".1 .5" pos="0 0 .5" euler="0 0 0" material="blue" mass="1" />
                    <geom name="end_mass" type="sphere" size=".25" pos="0 0 1" euler="0 0 0" material="blue" mass="1" />
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
viewer = mujoco_viewer.MujocoViewer(model, data)

for _ in range(10000):
    # Render forces
    viewer.show_actuator_forces(
        site_list=["site_1", "site_2", "site_3"],
        actuator_list=["pos_servo_1", "pos_servo_2", "pos_servo_3"],
        rgba_list=[1, 0.5, 1, 0.5],
        force_scale=0.05,
        arrow_radius=0.05,
        show_force_labels=True,
    )

    # step and render
    mujoco.mj_step(model, data)
    viewer.render()

# close
viewer.close()