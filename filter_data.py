import pandas as pd
import numpy as np
import glob

file_list = glob.glob("csv/*.csv")


def process_falling_data(file_path, target_count=15):
    df = pd.read_csv(file_path)

    # 1. Filter data di bawah 150 cm
    mask_below_150 = df['Height_cm'] <= 150
    if not mask_below_150.any():
        return None

    # Ambil index pertama saat di bawah 150
    start_idx = df[mask_below_150].index[0]

    # 2. titik terendah (sebelum memantul) setelah start_idx
    # Kita cari minimum height di sisa data tersebut
    falling_df = df.iloc[start_idx:]
    min_height_idx = falling_df['Height_cm'].idxmin()

    # Segment jatuh murni (dari 150 ke 0/min)
    segment = df.loc[start_idx:min_height_idx]

    # 3. ambil 10-15 sampel data secara merata
    if len(segment) > target_count:
        indices = np.linspace(0, len(segment) - 1, target_count).astype(int)
        subsampled_df = segment.iloc[indices]
    else:
        subsampled_df = segment

    return subsampled_df


for file in file_list:
    result = process_falling_data(file)
    if result is not None:
        result.to_csv(f"hasil_{file}", index=False)
        print(f"Selesai memproses {file} -> hasil_{file}")