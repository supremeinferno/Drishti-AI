"""Build your own currency dataset with the webcam.

    python collect_currency.py --label 500
    python collect_currency.py --label background   # no note in view, so the model can say "no note"

SPACE = save one photo, B = toggle burst mode (saves every 5th frame), Q = quit.
Move the note around: different angles, distances, lighting, folded, partly covered.
"""
import argparse
import os
import time

import cv2

import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="e.g. 10, 20, 50, 100, 200, 500, background")
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()

    folder = os.path.join(args.out, args.label)
    os.makedirs(folder, exist_ok=True)
    count = len(os.listdir(folder))
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    burst, frame_no = False, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        key = cv2.waitKey(1) & 0xFF
        save = key == ord(" ") or (burst and frame_no % 5 == 0)
        if save:
            cv2.imwrite(os.path.join(folder, f"{args.label}_{int(time.time() * 1000)}.jpg"), frame)
            count += 1
        if key == ord("b"):
            burst = not burst
        if key == ord("q"):
            break

        preview = frame.copy()
        status = f"{args.label}: {count} images" + ("  [BURST]" if burst else "")
        cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("collect currency", preview)

    cap.release()
    cv2.destroyAllWindows()
    print(f"saved {count} images in {folder}")


if __name__ == "__main__":
    main()
