import cv2
import pandas as pd
import os

# --- KONFIGURASI ---
FILE_NAME = "persatu"
NAMA_FILE_CSV = f'hasil_csv/{FILE_NAME}.csv'
JALUR_VIDEO = f"video/{FILE_NAME}.mp4"
FOLDER_OUTPUT = f"validasi_frame/{FILE_NAME}"
JUMLAH_SAMPEL = 15

if not os.path.exists(FOLDER_OUTPUT):
    os.makedirs(FOLDER_OUTPUT)


def validasi_frame_final():
    # 1. Cek status file video
    if not os.path.exists(JALUR_VIDEO):
        print(f"ERROR: File video tidak ditemukan di {os.path.abspath(JALUR_VIDEO)}")
        return

    # 2. Baca CSV dan ambil sampel frame
    try:
        df = pd.read_csv(NAMA_FILE_CSV)
        df_sampel = df.sample(n=min(JUMLAH_SAMPEL, len(df))).sort_values(by='Frame_Index')
        target_frames = df_sampel['Frame_Index'].astype(int).tolist()
    except Exception as e:
        print(f"ERROR saat membaca CSV: {e}")
        return

    # 3. Buka video dengan backend FFMPEG
    cap = cv2.VideoCapture(JALUR_VIDEO, cv2.CAP_FFMPEG)

    total_frame_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    print(f"--- INFO VIDEO ---")
    print(f"Total Frame di Video: {total_frame_video}")
    print(f"FPS Video: {fps_video}")
    print(f"Target Frame dari CSV: {target_frames}")
    print(f"------------------\n")

    if not cap.isOpened():
        print("Gagal membuka video!")
        return

    current_frame = 0
    success_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            if current_frame < max(target_frames):
                print(
                    f"Peringatan: Video berhenti mendadak di frame {current_frame}.")
            break

        current_frame += 1

        if current_frame in target_frames:
            baris_data = df[df['Frame_Index'] == current_frame]
            tinggi = baris_data['Height_cm'].values[0]

            label = f"F: {current_frame}"
            cv2.putText(frame, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            nama_gambar = os.path.join(FOLDER_OUTPUT, f"validasi_f{current_frame}.jpg")
            cv2.imwrite(nama_gambar, frame)

            print(f"[{success_count + 1}/{len(target_frames)}] Berhasil menyimpan: {nama_gambar}")
            success_count += 1

            # Berhenti jika sudah semua target diambil
            if success_count == len(target_frames):
                break

    cap.release()
    print(f"\nProses Selesai. Berhasil mengambil {success_count} gambar.")


if __name__ == "__main__":
    validasi_frame_final()