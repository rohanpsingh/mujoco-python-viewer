import mujoco
import mujoco_viewer

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option gravity="0 0 0" timestep="0.002" />
    <asset>
        <material name="blue_" rgba="0 0 1 1" />
        <material name="green" rgba="0 1 0 1" />
        <material name="red__" rgba="1 0 0 1" />
        <material name="white" rgba="1 1 1 1" />
    </asset>
    <worldbody>
        <geom type="plane" size="10 10 0.1" pos="0 0 -3" rgba=".9 0 0 1" />
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1" />
        <body name="link_1" pos="0 0 0">
            <geom type="cylinder" size=".2    2" pos="0 0 2" euler="0 0 0" material="red__" mass="0.8" />
            <geom type="cylinder" size=".25 .25" pos="0 0 4" euler="0 90 0" material="red__" mass="0.1" />
            <geom type="cylinder" size=".25 .25" pos="0 0 0" euler="0 90 0" material="red__" mass="0.1" />
            <body name="link_2" pos="0.5 0 0" euler="0 0 0">
                <joint name="hinge_1" type="hinge" pos="0 0 0" axis="1 0 0" /> <!--limited="true" range="-30 30" -->
                <geom type="cylinder" size=".2    2" pos="0 2 0" euler="90 0 0" material="blue_" mass="0.8" />
                <geom type="cylinder" size=".25 .25" pos="0 4 0" euler="0 90 0" material="blue_" mass="0.1" />
                <geom type="cylinder" size=".25 .25" pos="0 0 0" euler="0 90 0" material="blue_" mass="0.1" />
                <body name="link_3" pos="-0.5 4 0" euler="0 0 0">
                    <joint name="hinge_2" type="hinge" pos="0 0 0" axis="1 0 0" />
                    <geom type="cylinder" size=".2    2" pos="0 0 2" euler="0 0 0" material="green" mass="0.8" />
                    <geom type="cylinder" size=".25 .25" pos="0 0 0" euler="0 90 0" material="green" mass="0.1" />
                    <geom type="cylinder" size=".25 .25" pos="0 0 4" euler="0 90 0" material="green" mass="0.1" />
                    <body name="link_4" pos="0.5 0 4" euler="0 0 0">
                        <joint name="hinge_3" type="hinge" pos="0 0 0" axis="1 0 0" />
                        <geom type="cylinder" size=".2    2" pos="0 -2 0" euler="90 0 0" material="white" mass="0.8" />
                        <geom type="cylinder" size=".25 .25" pos="0 0 0" euler="0 90 0" material="white" mass="0.1" />
                        <geom type="cylinder" size=".25 .25" pos="0 -4 0" euler="0 90 0" material="white" mass="0.1" />
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    <equality>
        <connect name="kinematic_link" active="true" body1="link_1" body2="link_4" anchor="0 0 4" />
    </equality>
    <actuator>
        <motor name="hinge_1" joint="hinge_1" forcelimited="true" forcerange="-1000 1000" /> <!-- gear="1"  -->
        <position name="position_servo" joint="hinge_1" kp="100" />
        <velocity name="velocity_servo" joint="hinge_1" kv="100" />
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(MODEL_XML)
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)


# simulate and render
for _ in range(100000):
    mujoco.mj_step(model, data)
    viewer.render()

# close
viewer.close()
