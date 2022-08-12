# Viewer for MuJoCo in Python

Interactive renderer to use with the official Python bindings for MuJoCo 2.2.x

+ Develop along with [jx-mujoco-python-engine](https://github.com/jaku-jaku/jx-mujoco-python-engine)
+ A more efficient, robust and thread safe viewer with native MuJoCo 2.2.x with glfw
+ Works with Mac M1 with MuJoCo 2.2.x
## Points to ponder:
> This repo is/will-be a re-implementation of https://github.com/rohanpsingh/mujoco-python-viewer with custom features
> Additional credits: I do see a lot of similar structure as used in `mjviewer` from mujoco-py, so I will also give credits to https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjviewer.py
> Starting with version 2.1.2, MuJoCo comes with native Python bindings officially supported by the MuJoCo devs.  
> If you have been a user of `mujoco-py`, you might be looking to migrate.  
> Some pointers on migration are available [here](https://mujoco.readthedocs.io/en/latest/python.html#migration-notes-for-mujoco-py).

## Tested OS Platform:
- Mac M1
- Ubuntu 18.04

## Install
```sh
$ git clone https://github.com/jaku-jaku/jx-mujoco-python-viewer
$ cd jx-mujoco-python-viewer
$ pip install -e .
```

## Demo (WIP):
- https://github.com/UW-Advanced-Robotics-Lab/simulation-mujoco-summit-wam