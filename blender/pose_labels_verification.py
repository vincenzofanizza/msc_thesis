'''
Script to verify the generation of pose labels.

'''
import bpy

import numpy as np
import math
import mathutils
import sys
import os
import imp

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

import blender_utils as bu
imp.reload(bu)


bu.reset_viewport()
bu.add_primitive_cubes(verification_test = 'pose_labels')

camera_pose = {
    'location': mathutils.Vector((5., 0., 0.)),
    'rotation': mathutils.Euler(np.radians([0., 90., 90.]), 'ZYX')
}
rotation_intr = mathutils.Euler((np.radians(0.), np.radians(0.), np.radians(0.)), 'ZYX')
# METHOD USED TO DETERMINE EXTRINSIC ROTATIONS DOES NOT WORK - USE SCIPY
rotation_extr = mathutils.Euler((np.radians(0.), np.radians(0.), np.radians(0.)), 'XYZ')

camera = bu.place_camera()

camera = bu.change_camera_pose(camera, location = camera_pose['location'], rotation = camera_pose['rotation'], rot_type = 'euler')
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 3

scene = bpy.context.scene
scene.camera = camera

# Setup the rendering settings
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.film_transparent = True
