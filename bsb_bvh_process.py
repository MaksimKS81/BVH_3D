import os
import sys
import pandas as pd
import numpy as np
from bvh import Bvh  # type: ignore

from scipy.spatial.transform import Rotation as R


def bvh_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Read a BVH file and convert it into a pandas DataFrame containing:
      - Time (seconds)
      - Global joint coordinates (X, Y, Z) for each joint
      - Joint rotation angles (XYZ) for each joint
    """
    # Load and parse BVH
    with open(file_path, 'r') as f:
        mocap = Bvh(f.read())

    frame_time = mocap.frame_time
    joint_names = mocap.get_joints_names()
    # Identify the root joint explicitly
#    root_joint = joint_names[0]

    # 1. Rotation DataFrame
    channel_names = [f"{joint}_{ch}" for joint in joint_names
                     for ch in mocap.joint_channels(joint)]
    rot_df = pd.DataFrame(mocap.frames, columns=channel_names).astype(float)
    rot_df.insert(0, 'Time', rot_df.index * frame_time)

    # Vectorized coordinate extraction using NumPy batch operations
    # Convert frames to NumPy array: shape (n_frames, total_channels)
    frame_vals = np.array(mocap.frames, dtype=float)
    n_frames, total_ch = frame_vals.shape
    # Map joints to their channels and offsets
    channels_map = {j: mocap.joint_channels(j) for j in joint_names}
    offset_map = {j: np.array(mocap.joint_offset(j), dtype=float) for j in joint_names}
    # Build frame column mapping: list of (joint, channel)
    frame_cols = [(j, ch) for j in joint_names for ch in channels_map[j]]
    # Parent relationship from channel ordering and mocap hierarchy
    parent_map: dict[str, str | None] = {joint_names[0]: None}
    # Build parent map via AST: find parent of each joint
    for j in joint_names:
        node = mocap.get_joint(j)
        for child in getattr(node, 'children', []):
            if child.value and child.value[0] == 'JOINT':
                parent_map[child.value[1]] = j

    # Initialize storage
    global_rot: dict[str, np.ndarray] = {}
    global_pos: dict[str, np.ndarray] = {}
    # Compute per-joint transforms
    for joint in joint_names:
        # indices in frame_vals
        idxs = [i for i, (j, _) in enumerate(frame_cols) if j == joint]
        data = frame_vals[:, idxs]
        # Initialize local rotation and translation
        rot_local = np.repeat(np.eye(3)[None, :, :], n_frames, axis=0)
        trans_local = np.zeros((n_frames, 3), dtype=float)
        # Apply channels in order for all frames
        for k, ch in enumerate(channels_map[joint]):
            vals = data[:, k]
            if ch.endswith('rotation'):
                theta = np.deg2rad(vals)
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                # Build axis-specific rotation matrices
                R_frames = np.zeros((n_frames, 3, 3), dtype=float)
                if ch.startswith('X'):
                    R_frames[:, 0, 0] = 1
                    R_frames[:, 1, 1] = cos_t
                    R_frames[:, 1, 2] = -sin_t
                    R_frames[:, 2, 1] = sin_t
                    R_frames[:, 2, 2] = cos_t
                elif ch.startswith('Y'):
                    R_frames[:, 0, 0] = cos_t
                    R_frames[:, 0, 2] = sin_t
                    R_frames[:, 1, 1] = 1
                    R_frames[:, 2, 0] = -sin_t
                    R_frames[:, 2, 2] = cos_t
                else:  # Z
                    R_frames[:, 0, 0] = cos_t
                    R_frames[:, 0, 1] = -sin_t
                    R_frames[:, 1, 0] = sin_t
                    R_frames[:, 1, 1] = cos_t
                    R_frames[:, 2, 2] = 1
                # Batch multiply: rot_local = rot_local @ R_frames
                rot_local = np.einsum('nij,njk->nik', rot_local, R_frames)
            else:
                # Position channels
                axis = ['X', 'Y', 'Z'].index(ch[0])
                trans_local[:, axis] = vals
        # Determine global transform
        parent = parent_map[joint]
        if parent is None:
            # root joint
            pos_global = (trans_local if any('position' in c for c in channels_map[joint])
                          else np.repeat(offset_map[joint][None, :], n_frames, axis=0))
            rot_global = rot_local
        else:
            # offset-based position if no position channels
            if any('position' in c for c in channels_map[joint]):
                pos_global = global_pos[parent] + trans_local
            else:
                # parent_rot @ offset
                pos_global = global_pos[parent] + np.einsum('nij,j->ni', global_rot[parent], offset_map[joint])
            # combine rotations
            rot_global = np.einsum('nij,njk->nik', global_rot[parent], rot_local)
        global_pos[joint] = pos_global
        global_rot[joint] = rot_global

    # Stack positions for DataFrame
    coord_array = np.hstack([global_pos[j] for j in joint_names])
    time_col = np.arange(n_frames) * frame_time
    coord_cols = [f"{j}_{axis}" for j in joint_names for axis in ['X', 'Y', 'Z']]
    coord_df = pd.DataFrame(np.column_stack((time_col, coord_array)),
                            columns=['Time'] + coord_cols)
     
    # 5. Merge coordinates and rotations
    merged_df = pd.merge(coord_df, rot_df, on='Time')
    return merged_df


def compute_relative_cardan_angles(df: pd.DataFrame,
                                   segment1: str,
                                   segment2: str,
                                   sequence: str = 'XYZ',
                                   degrees: bool = True) -> pd.DataFrame:
    """
    For each row in the DataFrame, compute the relative 3-component Cardan (Euler) angles
    between the local frames of segment1 and segment2.

    Parameters:
    - df: DataFrame with columns '<segment>_Xrotation', '<segment>_Yrotation', '<segment>_Zrotation'
    - segment1: name of the parent segment (e.g. 'RightArm')
    - segment2: name of the child segment (e.g. 'RightForeArm')
    - sequence: rotation sequence (default 'XYZ')
    - degrees: whether angles are in degrees (default True)

    Returns:
    - DataFrame with added columns for relative angles:
      'rel_<segment1>_<segment2>_X', 'rel_<segment1>_<segment2>_Y', 'rel_<segment1>_<segment2>_Z'
    """
    rel_angles = []
    for _, row in df.iterrows():
        # Extract Euler angles
        angles1 = [row[f"{segment1}_{axis}rotation"] for axis in sequence]
        angles2 = [row[f"{segment2}_{axis}rotation"] for axis in sequence]
        # Create Rotation objects
        r1 = R.from_euler(sequence, angles1, degrees=degrees)
        r2 = R.from_euler(sequence, angles2, degrees=degrees)
        # Relative rotation: r_rel = inverse(r1) * r2
        r_rel = r1.inv() * r2
        # Decompose back to Euler
        rel = r_rel.as_euler(sequence, degrees=degrees)
        rel_angles.append(rel)

    # Build DataFrame of relative angles
    rel_cols = [f"rel_{segment1}_{segment2}_{axis}" for axis in sequence]
    rel_df = pd.DataFrame(rel_angles, columns=rel_cols, index=df.index)
    # Concatenate
    return pd.concat([df, rel_df], axis=1)


def angle_between_vectors_from_three_points(df, 
    p1a, p1b,  # e.g. ['RightShoulder_X', 'RightShoulder_Y', 'RightShoulder_Z'], ['RightArm_X', 'RightArm_Y', 'RightArm_Z']
    p2a, p2b   # e.g. ['RightForeArm_X', 'RightForeArm_Y', 'RightForeArm_Z'], ['RightHand_X', 'RightHand_Y', 'RightHand_Z']
):
    """
    Calculate the angle (degrees) between two vectors defined by two pairs of points for each row in df.
    Returns a pandas Series of angles (degrees).
    """
    # Extract coordinates as arrays
    v1 = df[[c for c in p1b]].values - df[[c for c in p1a]].values  # Vector 1: p1b - p1a
    v2 = df[[c for c in p2b]].values - df[[c for c in p2a]].values  # Vector 2: p2b - p2a

    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=1, keepdims=True)

    # Compute dot product and angle
    dot = np.einsum('ij,ij->i', v1_norm, v2_norm)
    # Clamp dot to [-1, 1] to avoid numerical errors
    dot = np.clip(dot, -1.0, 1.0)
    angles_rad = np.arccos(dot)
    angles_deg = np.degrees(angles_rad)
    
    return angles_deg



if __name__ == "__main__":
    
    data_folder = "DATA/"
    file_name = "unknown.bvh"
    with open(data_folder + file_name, 'r') as f:
        mocap = Bvh(f.read())
        
        
    df = bvh_to_dataframe(data_folder + file_name)

    # Debug: inspect extract_positions output for first frame
    m = Bvh(open(os.path.join('DATA','unknown.bvh'),'r').read())
    first_frame = m.frames[0]
    pos = bvh_to_dataframe(os.path.join('DATA','unknown.bvh'))  # or call extract_positions directly if exposed
    print("Extracted positions dict keys:", pos.keys())
    #print({k: v for k,v in pos.items() if not np.isnan(v).all()})

# Save DataFrame to CSV
#csv_path = 'bvh_output.csv'
#df.to_csv(csv_path, index=False)

# Read DataFrame from CSV
#loaded_df = pd.read_csv(csv_path)
#print(loaded_df.head())
