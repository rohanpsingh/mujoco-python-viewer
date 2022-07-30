import mujoco
import mujoco_viewer
import PIL.Image

model = mujoco.MjModel.from_xml_path('humanoid.xml')
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data, if_on_scrn=False)
mujoco.mj_step(model, data)
viewer.process_safe()
viewer.render_safe()
    
# img = viewer.read_pixels(camid=2)
# img = PIL.Image.fromarray(img)
# img.show()
