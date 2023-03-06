import bpy

import numpy as np
import math
import mathutils

from scipy.spatial.transform import Rotation


def reset_viewport():
    '''
    Delete all objects added to the viewport in a previous session.
    
    '''
    bpy.ops.object.select_all(action = 'SELECT')
    bpy.ops.object.delete(use_global = False)

def add_primitive_cubes(verification_test = None):
    '''
    Add cubes to the viewport required for verification.
    
    '''
    if verification_test == 'image_rendering':
        # Create central large cube
        bpy.ops.mesh.primitive_cube_add(location = (0., 0., 0.), size = 4)

        # Create smaller cubes
        bpy.ops.mesh.primitive_cube_add(location = (2.5, -1.5, -1.5), size = 1)
        bpy.ops.mesh.primitive_cube_add(location = (-1., 3., -1.), size = 2)
    else:
        raise ValueError('verification test not selected')

def quat2eul(quaternion = None, order = 'ZYX'):
    '''
    Convert quaternion (scalar-first notation) to euler angles (in radians) assuming a specified rotation sequence.

    '''
    return quaternion.to_euler(order)

def place_camera(location = None, rotation = None, rot_type = 'euler'):
    '''
    Create and add a camera to the scene with a certain location and rotation.

    '''
    # Create camera
    bpy.ops.object.camera_add()
    camera = bpy.data.objects['Camera']

    if rot_type == 'quaternion':
        # Quaternions are assumed in a scalar-first notation
        rotation = quat2eul(rotation)
    elif rot_type != 'euler':
        raise ValueError('camera rotation type is unknown')
    
    camera.location = location
    camera.rotation_mode = 'ZYX'
    camera.rotation_euler = np.array([rotation.x, rotation.y, rotation.z])

    return camera


camera_poses = {
    'location': np.array([[20., 0., 0.], [0., 20., 0.], [-20., 0., 0.], [0., -20., 0.], [0., 0., 20.], [0., 0., 20.], [0., 0., -20.], [0., 0., -20.], [10., -20., 0.]]),
    'rotation': np.array([[0.5, 0.5, 0.5, 0.5], [0., 0.0, -0.707107, -0.707107], [0.5, 0.5, -0.5, -0.5], [0.707107, 0.707107, 0., 0.], [0.707107, 0., 0., 0.707107], [1., 0., 0., 0.], [0., 0.707107, -0.707107, 0.], [0., 0.382683, -0.923880, 0.], [0.707107, 0.707107, 0., 0.]])
}


reset_viewport()
add_primitive_cubes(verification_test = 'image_rendering')

for i in range(len(camera_poses['location'])):
    camera = place_camera(location = camera_poses['location'][i], rotation = mathutils.Quaternion(camera_poses['rotation'][i]), rot_type = 'quaternion')
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

