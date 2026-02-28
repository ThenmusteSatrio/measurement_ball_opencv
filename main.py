import cv2
import numpy as np
import csv

FILE_NAME = "persatu"
VIDEO_PATH = f'video/{FILE_NAME}.mp4'
FILE_OUTPUT_CSV = f'csv/{FILE_NAME}.csv'
VIDEO_OUTPUT_PATH = f'output/{FILE_NAME}.mp4'

LOWER_ORANGE = np.array([5, 150, 150])
UPPER_ORANGE = np.array([25, 255, 255])

ROI_X_START = 250
ROI_X_END = 650


PT_150 = (455, 155)
PT_0 = (520, 1720)


CUSTOM_SCALES = {
    150: 155,
    140: 280,
    130: 405,
    120: 525,
    110: 640,
    100: 750,
    90: 860,
    80: 970,
    70: 1075,
    60: 1175,
    50: 1270,
    40: 1367,
    30: 1460,
    20: 1550,
    10: 1640,
    0: 1720
}


def get_height_from_y(y_pixel):
    sorted_items = sorted(CUSTOM_SCALES.items(), key=lambda x: x[1])

    if y_pixel <= sorted_items[0][1]: return sorted_items[0][0]
    if y_pixel >= sorted_items[-1][1]: return sorted_items[-1][0]

    for i in range(len(sorted_items) - 1):
        cm_top, y_top = sorted_items[i]
        cm_bot, y_bot = sorted_items[i + 1]

        if y_top <= y_pixel <= y_bot:
            # Kalkulasi persentase posisi di dalam segmen tersebut
            segment_ratio = (y_pixel - y_top) / (y_bot - y_top)
            return cm_top - (segment_ratio * (cm_top - cm_bot))
    return 0


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, video_fps, (width, height))

    print(f"FPS Terdeteksi: {video_fps}")
    data_log = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        millis = (frame_count / video_fps) * 1000

        # 1. Gambar Garis Biru Utama
        cv2.line(frame, PT_150, PT_0, (255, 0, 0), 2)

        # 2. Gambar Penggaris Berdasarkan Mapping CUSTOM_SCALES
        for cm, y_pos in CUSTOM_SCALES.items():
            line_ratio = (y_pos - PT_150[1]) / (PT_0[1] - PT_150[1])
            curr_x = int(PT_150[0] + line_ratio * (PT_0[0] - PT_150[0]))

            color_mark = (0, 0, 255) if cm in [0, 150] else (255, 255, 255)
            cv2.line(frame, (curr_x - 15, y_pos), (curr_x, y_pos), color_mark, 2)
            cv2.putText(frame, f"{cm}", (curr_x - 45, y_pos + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_mark, 1)

        # 3. ROI & Tracking
        roi = frame[:, ROI_X_START:ROI_X_END]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.putText(frame, f"Time: {millis:.2f} ms", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 50:
                (x_c, y_c), radius = cv2.minEnclosingCircle(c)
                center_f = (int(x_c + ROI_X_START), int(y_c))

                # TINGGI menggunakan fungsi mapping non-linear
                height_cm = get_height_from_y(center_f[1])

                if -5 <= height_cm <= 155:
                    data_log.append([frame_count, round(millis, 2), round(height_cm, 2)])
                    text_x = center_f[0] + int(radius) + 10

                    cv2.circle(frame, center_f, int(radius), (0, 255, 0), 3)
                    cv2.putText(frame, f"Frame: {frame_count}", (text_x, int(y_c) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"{height_cm:.1f} cm", (text_x, int(y_c) + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out_video.write(frame)
        cv2.imshow('Tracking Precision - Adjusted Scale', frame)

        key = cv2.waitKey(500)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord(' '):
            cv2.waitKey(0)

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    with open(FILE_OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame_Index', 'Time_ms', 'Height_cm'])
        writer.writerows(data_log)
    print(f"Analisis Selesai! CSV: {FILE_OUTPUT_CSV}")


if __name__ == "__main__":
    main()