import pandas as pd
import numpy as np
import glob
import os

file_list = glob.glob("csv/*.csv")


def process_falling_data(file_path, target_count=15):
    df = pd.read_csv(file_path)

    df['v_temp'] = -df['Height_cm'].diff()
    mask_moving = df['v_temp'] > 0.5
    if not mask_moving.any(): return None
    start_idx = df[mask_moving].index[0]

    h_start = np.floor(df.loc[start_idx, 'Height_cm'])

    mask_floor = df['Height_cm'] < 2.0
    if mask_floor.any():
        end_idx = df[mask_floor].index[0] - 1
    else:
        end_idx = df['Height_cm'].idxmin()

    if end_idx <= start_idx: end_idx = len(df) - 1

    segment = df.loc[start_idx:end_idx].copy()
    segment = segment.drop_duplicates(subset=['Height_cm'], keep='first')

    if len(segment) > target_count:
        indices = np.linspace(0, len(segment) - 1, target_count).astype(int)
        subsampled_df = segment.iloc[indices].copy()
    else:
        subsampled_df = segment.copy()

    subsampled_df['t (s)'] = (subsampled_df['Time_ms'] - subsampled_df['Time_ms'].iloc[0]) / 1000.0
    subsampled_df['t^2 (s^2)'] = subsampled_df['t (s)'] ** 2

    subsampled_df['h (cm)'] = h_start - subsampled_df['Height_cm']

    subsampled_df['h (cm)'] = subsampled_df['h (cm)'].clip(lower=0)

    subsampled_df['2h (cm)'] = 2 * subsampled_df['h (cm)']
    subsampled_df = subsampled_df.round(4)

    return subsampled_df

for file in file_list:
    result = process_falling_data(file)
    if result is not None:
        nama_file = os.path.basename(file)
        result.to_csv(f"hasil_csv/{nama_file}", index=False)
        print(f"Selesai memproses {file} -> hasil_{nama_file}")