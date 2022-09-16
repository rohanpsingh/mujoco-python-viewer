import mujoco
import mujoco_viewer
import math

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


# Initialize the graph lines
viewer.add_graph_line(line_name="line_1")
viewer.add_graph_line(line_name="downscaled_force_sensor")
viewer.add_graph_line(line_name="position_sensor")

# Initialize the Legend
viewer.show_graph_legend(show_legend=True)

# For a time-based graph, 
# x_axis_time is the total time that you want your viewing window to see
# It needs to be set to grater than [model.opt.timestep*50]
# If you want to over ride that, change the "override" parameter to True
viewer.set_grid_divisions(x_div=5,
                          y_div=4,
                          x_axis_time=model.opt.timestep * 1000,
                          override=True)

for i in range(10000):
    # Render forces
    viewer.show_actuator_forces(
        site_list=["site_1", "site_2", "site_3"],
        actuator_list=["pos_servo_1", "pos_servo_2", "pos_servo_3"],
        rgba_list=[1, 0.5, 1, 0.5],
        force_scale=0.05,
        arrow_radius=0.05,
        show_force_labels=True,
    )

    viewer.update_graph_line(line_name="line_1", line_data=math.sin(i / 10.0))
    viewer.update_graph_line(line_name="downscaled_force_sensor", line_data=data.actuator_force[0]/100.0)
    viewer.update_graph_line(line_name="position_sensor", line_data=data.sensordata[0])

    # step and render
    mujoco.mj_step(model, data)
    viewer.render()
    if not viewer.is_alive:
        break

# close
viewer.close()
