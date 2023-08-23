import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('humanoid.xml')
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

viewer.add_line_to_fig(line_name="root-pos-x", fig_idx = 0)
viewer.add_line_to_fig(line_name="root-pos-z", fig_idx = 0)
viewer.add_line_to_fig(line_name="right_ankle_y", fig_idx = 1)

# user has access to mjvFigure
fig = viewer.figs[0]
fig.title = "Root Position"
fig.flg_legend = True
fig.xlabel = "Timesteps"
fig.figurergba[0] = 0.2
fig.figurergba[3] = 0.2
fig.gridsize[0] = 5
fig.gridsize[1] = 5

fig = viewer.figs[1]
fig.title = "Joint position"
fig.flg_legend = True
fig.figurergba[0] = 0.2
fig.figurergba[3] = 0.2

# simulate and render
for _ in range(100000):
    viewer.add_data_to_line(line_name="root-pos-x", line_data=data.qpos[0], fig_idx=0)
    viewer.add_data_to_line(line_name="root-pos-z", line_data=data.qpos[2], fig_idx=0)
    viewer.add_data_to_line(line_name="right_ankle_y", line_data=data.qpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_ankle_y")], fig_idx=1)

    mujoco.mj_step(model, data)
    viewer.render()
    if not viewer.is_alive:
        break

# close
viewer.close()

