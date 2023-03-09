'''
Script to verify Blender's rendering capabilities.

'''

import bpy

import numpy as np
import math
import mathutils
import sys
import os

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

import blender_utils as bu

import imp
imp.reload(bu)


camera_poses = {
    'location': np.array([[20., 0., 0.], [0., 20., 0.], [-20., 0., 0.], [0., -20., 0.], [0., 0., 20.], [0., 0., 20.], [0., 0., -20.], [0., 0., -20.], [10., -20., 0.]]),
    'rotation': np.array([[0.5, 0.5, 0.5, 0.5], [0., 0.0, -0.707107, -0.707107], [0.5, 0.5, -0.5, -0.5], [0.707107, 0.707107, 0., 0.], [0.707107, 0., 0., 0.707107], [1., 0., 0., 0.], [0., 0.707107, -0.707107, 0.], [0., 0.382683, -0.923880, 0.], [0.707107, 0.707107, 0., 0.]])
}

bu.reset_viewport()
bu.add_primitive_cubes(verification_test = 'image_rendering')

for i in range(camera_poses['location'].shape[0]):
    camera = bu.place_camera(location = mathutils.Vector(camera_poses['location'][i]), rotation = mathutils.Quaternion(camera_poses['rotation'][i]), rot_type = 'quaternion')
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 20
    
    scene = bpy.context.scene
    scene.camera = camera

    # Setup the rendering settings
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = True
    
    # Setup rendering filepath
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = "//image_rendering_verification/pose_{num}.png".format(num = i+1)

    # Render
    bpy.ops.render.render(write_still = 1)

    bpy.ops.object.select_all(action = 'DESELECT')
    bpy.data.objects['Camera'].select_set(True)
    bpy.ops.object.delete() 
    
