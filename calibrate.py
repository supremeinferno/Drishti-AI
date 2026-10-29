"""Find your webcam's focal length (in pixels) for the pinhole distance formula.

Stand at a measured distance from the camera (e.g. 2 m, full body visible) and press SPACE
a few times. focal = pixel_height * distance / real_height. Paste the result into config.py.

    python calibrate.py --distance 2.0 --height 1.70
"""
import argparse
import statistics

import cv2

import config
from detector import Detector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distance", type=float, required=True, help="metres from camera to person")
    p.add_argument("--height", type=float, required=True, help="person's real height in metres")
    args = p.parse_args()

    detector = Detector(config.YOLO_WEIGHTS, config.CONF_THRESHOLD)
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    samples = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        people = [d for d in detector(frame) if d.label == "person"]
        person = max(people, key=lambda d: d.box[3] - d.box[1], default=None)
        if person:
            x1, y1, x2, y2 = person.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and person:
            samples.append((person.box[3] - person.box[1]) * args.distance / args.height)
            print(f"sample {len(samples)}: focal = {samples[-1]:.1f}")
        if key == ord("q"):
            break

        cv2.putText(frame, f"samples: {len(samples)}  SPACE=capture  Q=done", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("calibrate", frame)

    cap.release()
    cv2.destroyAllWindows()
    if samples:
        print(f"\nFOCAL_LENGTH_PX = {statistics.median(samples):.1f}   # paste into config.py")


if __name__ == "__main__":
    main()
