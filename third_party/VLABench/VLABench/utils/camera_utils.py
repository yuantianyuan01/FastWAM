import numpy as np

def get_camera_target(pos, xyaxes, distance=1.0):
   """
   Compute the target point for a camera given its position, orientation in 6D, and distance.
   """
   R = rotation_6d_to_matrix(xyaxes)
   
   # Camera's Z-axis direction (optical axis direction)
   # Note: Could be R[:, 2] or -R[:, 2] depending on coordinate system convention
   forward = -R[:, 2]  # or -R[:, 2], depends on coordinate system
   
   target = pos + forward * distance
   return target

def rotation_6d_to_matrix(d6):
   """
   convert a 6D representation of rotation to a rotation matrix.
   """
   a1 = d6[:3]
   a2 = d6[3:]
   
   b1 = a1 / np.linalg.norm(a1)
   b2 = a2 - np.dot(b1, a2) * b1
   b2 = b2 / np.linalg.norm(b2)
   b3 = np.cross(b1, b2)
   
   return np.stack([b1, b2, b3], axis=1)

def move_camera_keep_target(original_pos, original_xyaxes, new_pos, target_distance=1.0):
   """
   Move camera to new position while keeping the target point unchanged
   
   Args:
       original_pos: Original camera position [x, y, z]
       original_xyaxes: Original 6D rotation vector
       new_pos: New camera position [x, y, z]
       target_distance: Target point distance (used to calculate original target point)
   
   Returns:
       new_xyaxes: New 6D rotation vector
   """
   # 1. Calculate original target point
   target = get_camera_target(original_pos, original_xyaxes, target_distance)
   
   # 2. Calculate new orientation
   new_xyaxes = look_at_to_6d(new_pos, target)
   
   return new_xyaxes

def look_at_to_6d(camera_pos, target_pos, up_vector=np.array([0, 0, 1])):
   """
   Calculate 6D rotation vector from camera position and target point
   
   Args:
       camera_pos: Camera position
       target_pos: Target point position
       up_vector: Up direction vector (world coordinate system)
   
   Returns:
       6D rotation vector
   """
   # Calculate forward vector (Z-axis)
   forward = target_pos - camera_pos
   forward = forward / np.linalg.norm(forward)
   
   # Calculate right vector (X-axis)
   right = np.cross(forward, up_vector)
   right = right / np.linalg.norm(right)
   
   # Calculate up vector (Y-axis)
   up = np.cross(right, forward)
   
   # Construct 6D vector (adjust according to your coordinate system convention)
   # If Z-axis points forward:
   xyaxes = np.concatenate([right, up])
   
   # If Z-axis points backward (OpenCV convention):
   # xyaxes = np.concatenate([right, -up])
   
   return xyaxes

def orbital_camera_movement(original_pos, 
                           original_xyaxes, 
                           angle, 
                           axis='y',
                           target_distance=1.0):
   """
   Perform orbital movement around the target point
   """
   target = get_camera_target(original_pos, original_xyaxes, distance=target_distance)
   
   # Calculate relative position
   relative_pos = original_pos - target
   
   # Rotation matrix
   if axis == 'y':
       rotation_matrix = np.array([
           [np.cos(angle), 0, np.sin(angle)],
           [0, 1, 0],
           [-np.sin(angle), 0, np.cos(angle)]
       ])
   elif axis == 'x':
       rotation_matrix = np.array([
           [1, 0, 0],
           [0, np.cos(angle), -np.sin(angle)],
           [0, np.sin(angle), np.cos(angle)]
       ])
   
   # New relative position
   new_relative_pos = rotation_matrix @ relative_pos
   new_pos = target + new_relative_pos
   
   # Calculate new 6D vector
   new_xyaxes = look_at_to_6d(new_pos, target)
   
   return new_pos, new_xyaxes

def translate_camera_keep_target(original_pos, original_xyaxes, translation, target_distance=1.0):
   """
   Translate camera while keeping the target point unchanged
   """
   new_pos = original_pos + translation
   new_xyaxes = move_camera_keep_target(
       original_pos, original_xyaxes, new_pos, target_distance=target_distance
   )
   return new_pos, new_xyaxes