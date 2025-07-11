import pandas as pd
from pathlib import Path
import ezc3d
import numpy as np

def process_basketball_data(file_name):
    """Load a C3D file from the DATA folder by filename, compute angles and velocities, return selected DataFrame."""
    # build full path to the c3d in DATA folder
    c3d = ezc3d.c3d(str(file_name))
    print(f"Processing file: {file_name}")

    # get labels + point data
    raw_labels = c3d['parameters']['POINT']['LABELS']['value']
    labels = [lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in raw_labels]
    points = c3d['data']['points']     # shape: (4, n_markers, n_frames)
    coords = points[:3, :, :]          # drop residual row

    # select only markers starting with 'player'
    idx = [i for i, name in enumerate(labels) if name.startswith('player')]
    player_names = [labels[i] for i in idx]

    # reshape to (n_frames, n_players, 3)
    player_data = coords[:, idx, :].transpose(2, 1, 0)
    n_frames, n_players, _ = player_data.shape

    # flatten to 2D
    flat = player_data.reshape(n_frames, n_players * 3)
    axes = ['X','Y','Z']
    cols = [f"{name}_{axis}" for name in player_names for axis in axes]
    df = pd.DataFrame(flat, columns=cols)

    # filter desired markers
    markers = ['RELB','RSHO','RWRA','LELB','LSHO','LWRA','LFHD','RFHD']
    df_selected = df.filter(regex='|'.join(markers)).copy()

    # compute right elbow angle
    e = df_selected[['player:RELB_X','player:RELB_Y','player:RELB_Z']].values
    s = df_selected[['player:RSHO_X','player:RSHO_Y','player:RSHO_Z']].values
    w = df_selected[['player:RWRA_X','player:RWRA_Y','player:RWRA_Z']].values
    v1 = s - e; v2 = w - e
    dot = np.einsum('ij,ij->i', v1, v2)
    norm1 = np.linalg.norm(v1, axis=1); norm2 = np.linalg.norm(v2, axis=1)
    df_selected['REL_ANG'] = np.degrees(np.arccos(dot / (norm1 * norm2)))

    # compute left elbow angle
    e_L = df_selected[['player:LELB_X','player:LELB_Y','player:LELB_Z']].values
    s_L = df_selected[['player:LSHO_X','player:LSHO_Y','player:LSHO_Z']].values
    w_L = df_selected[['player:LWRA_X','player:LWRA_Y','player:LWRA_Z']].values
    v1_L = s_L - e_L; v2_L = w_L - e_L
    dot_L = np.einsum('ij,ij->i', v1_L, v2_L)
    norm1_L = np.linalg.norm(v1_L, axis=1); norm2_L = np.linalg.norm(v2_L, axis=1)
    df_selected['LEL_ANG'] = np.degrees(np.arccos(dot_L / (norm1_L * norm2_L)))

    # compute velocities for wrists
    pos_R = df_selected[['player:RWRA_X','player:RWRA_Y','player:RWRA_Z']].values
    vel_R = np.linalg.norm(np.diff(pos_R, axis=0), axis=1)
    vel_R = np.insert(vel_R, 0, 0)
    df_selected['RWRA_V'] = vel_R

    pos_L = df_selected[['player:LWRA_X','player:LWRA_Y','player:LWRA_Z']].values
    vel_L = np.linalg.norm(np.diff(pos_L, axis=0), axis=1)
    vel_L = np.insert(vel_L, 0, 0)
    df_selected['LWRA_V'] = vel_L

    # compute angular velocities for elbow angles
    df_selected['LEL_ANG_V'] = df_selected['LEL_ANG'].diff()
    df_selected['REL_ANG_V'] = df_selected['REL_ANG'].diff()

    # fill first NaN in angular velocities by backfilling
    df_selected['LEL_ANG_V'].fillna(method='bfill', inplace=True)
    df_selected['REL_ANG_V'].fillna(method='bfill', inplace=True)

    return df_selected

def find_wrist_head_threshold_indices(df, markers=['player:LWRA_Z', 'player:RWRA_Z', 'player:LFHD_Z']):
    """Return first and last frame indices where both LWRA_Z and RWRA_Z exceed LFHD_Z."""
    mask = (
        (df[markers[0]] > df[markers[2]]) &
        (df[markers[1]] > df[markers[2]])    
        )
    true_idx = np.where(mask)[0]
    if true_idx.size == 0:
        return None, None
    return int(true_idx[0]), int(true_idx[-1])

def find_all_wrist_head_threshold_intervals(df, markers=['player:LWRA_Z', 'player:RWRA_Z', 'player:LFHD_Z']):
    """
    Return a list of (start_idx, end_idx) for all contiguous intervals where both LWRA_Z and RWRA_Z exceed LFHD_Z.
    """
    mask = (
        (df[markers[0]] > df[markers[2]]) &
        (df[markers[1]] > df[markers[2]])
    ).values

    intervals = []
    in_interval = False
    for i, val in enumerate(mask):
        if val and not in_interval:
            start = i
            in_interval = True
        elif not val and in_interval:
            end = i - 1
            intervals.append((start, end))
            in_interval = False
    if in_interval:
        intervals.append((start, len(mask) - 1))
    return intervals

# function to find maximum angular velocity indices
def find_max_angle_velocity_indices(df, markers=['REL_ANG_V', 'LEL_ANG_V']):
    """Return indices of maximum angular velocity for right and left elbows."""
    idx_rel = df[markers[0]].idxmax()
    idx_lel = df[markers[1]].idxmax()
    return int(idx_rel), int(idx_lel)




if __name__ == "__main__":
    # example usage
    data_dir = Path(__file__).parent / "DATA"
    file_name = data_dir / "124_03.c3d"
    if not file_name.exists():
        raise FileNotFoundError(f"File {file_name} does not exist. Please check the path.")
    df = process_basketball_data(file_name)
    print(df.head())