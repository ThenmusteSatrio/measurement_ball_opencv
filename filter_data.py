import pandas as pd
import numpy as np
import glob
import os

file_list = glob.glob("csv/*.csv")

def process_falling_data(file_path, target_count=15):
    df = pd.read_csv(file_path)

    mask_150 = df['Height_cm'] == 150
    if mask_150.any():
        start_idx = df[mask_150].index[-1]
    else:
        mask_below_150 = df['Height_cm'] < 150
        if not mask_below_150.any():
            return None
        start_idx = df[mask_below_150].index[0]

    mask_0 = df['Height_cm'] == 0
    if mask_0.any():
        end_idx = df[mask_0].index[0]
    else:
        end_idx = df.loc[start_idx:, 'Height_cm'].idxmin()

    segment = df.loc[start_idx:end_idx].copy()

    segment = segment.drop_duplicates(subset=['Height_cm'], keep='first')

    if len(segment) > target_count:
        indices = np.linspace(0, len(segment) - 1, target_count).astype(int)
        subsampled_df = segment.iloc[indices]
    else:
        subsampled_df = segment

    return subsampled_df

for file in file_list:
    result = process_falling_data(file)
    if result is not None:
        nama_file = os.path.basename(file)
        result.to_csv(f"hasil_csv/{nama_file}", index=False)
        print(f"Selesai memproses {file} -> hasil_{nama_file}")