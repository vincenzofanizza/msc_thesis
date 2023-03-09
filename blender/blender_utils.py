'''
Script containing utilities to work in Blender.
It is recommended to run this script using the Blender's Python console rather than from a terminal.

'''
import bpy

import numpy as np
import math
import mathutils


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
    elif verification_test == 'pose_labels':
        # Create one central cube
        bpy.ops.mesh.primitive_cube_add(location = (0., 0., 0.), size = 1)
    else:
        raise ValueError('verification test not selected')

def quat2eul(quaternion = None, order = 'ZYX'):
    '''
    Convert quaternion instance (scalar-first notation) into euler-angle instance (in radians) assuming a specified rotation sequence.

    '''
    return quaternion.to_euler(order)

def eul2quat(euler_angles = None):
    '''
    Convert euler-angle instance (in radians) into quaternion instance (scalar-first notation).

    '''
    return euler_angles.to_quaternion()

def place_camera(location = mathutils.Vector((0., 0., 0.)), rotation = mathutils.Euler((0., 0., 0.), 'ZYX'), rot_type = 'euler'):
    '''
    Create and add a camera to the scene with a certain location and rotation.

    '''
    # Create camera
    bpy.ops.object.camera_add()
    camera = bpy.data.objects['Camera']
    
    camera.location = location

    if rot_type == 'quaternion':
        # Quaternions are assumed in a scalar-first notation
        camera.rotation_mode = 'QUATERNION'
        camera.rotation_quaternion = rotation
    elif rot_type == 'euler':
        camera.rotation_mode = rotation.order
        camera.rotation_euler = rotation
    elif rot_type != 'euler':
        raise ValueError('camera rotation type is unknown')

    return camera

def change_camera_pose(camera, location = mathutils.Vector((0., 0., 0.)), rotation = mathutils.Euler((0., 0., 0.), 'ZYX'), rot_type = 'euler'):
    '''
    Change the camera pose to the specified location and rotation wrt the world frame.
    
    '''
    camera.location = location
    
    if rot_type == 'quaternion':
        # Quaternions are assumed in a scalar-first notation
        camera.rotation_mode = 'QUATERNION'
        camera.rotation_quaternion = rotation
    elif rot_type == 'euler':
        camera.rotation_mode = rotation.order
        camera.rotation_euler = rotation
    elif rot_type != 'euler':
        raise ValueError('camera rotation type is unknown')
    
    return camera
