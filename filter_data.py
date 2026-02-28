import pandas as pd
import numpy as np
import glob
import os

# Mencari semua file di dalam folder csv/
file_list = glob.glob("csv/*.csv")

def process_falling_data(file_path, target_count=15):
    # 1. Baca data
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(subset=['Height_cm'], keep='last')

    # 3. Filter data di bawah atau sama dengan 150 cm
    mask_below_150 = df['Height_cm'] <= 150
    if not mask_below_150.any():
        return None

    # Ambil index pertama saat mulai masuk area 150 ke bawah
    start_idx_label = df[mask_below_150].index[0]

    # 4. Cari titik terendah (dasar/sebelum memantul)
    falling_df = df.loc[start_idx_label:]
    min_height_idx_label = falling_df['Height_cm'].idxmin()

    # Potong segmen jatuh murni (150 -> 0)
    segment = df.loc[start_idx_label:min_height_idx_label]

    # 5. sampel 10-15 data secara merata
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