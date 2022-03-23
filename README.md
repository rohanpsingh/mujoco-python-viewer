# Viewer for MuJoCo in Python

This is a `mujoco-py` style non-interactive viewer for using with the official Python bindings for MuJoCo.

Starting with version 2.1.2, MuJoCo comes with native Python bindings officially supported by the MuJoCo devs.  

If you have been a user of `mujoco-py`, you might be looking to migrate.  
Some pointers on migration are available [here](https://mujoco.readthedocs.io/en/latest/python.html#migration-notes-for-mujoco-py).

# Install
```sh
$ git clone https://github.com/rohanpsingh/mujoco-python-viewer
$ cd mujoco-python-viewer
$ pip install .
```

# Usage

```py
import mujoco
import mujoco_viewer

model = mujoco.Mjodel.from_xml_path('humanoid.xml')
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(100000):
    mujoco.mj_step(model, data)
    viewer.render()
```
Press `ESC` to quit.  
Other key bindings are shown in the overlay menu (almost similar to `mujoco-py`).
