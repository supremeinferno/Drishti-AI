"""Drishti: real-time obstacle + currency assistant for visually impaired people.

    python main.py                # webcam, pinhole distance
    python main.py --depth        # use the Depth Anything V2 neural depth model
    python main.py --source walk.mp4 --mute

Keys:  C = read the currency note held in front of the camera,  Q = quit
"""
import argparse
import os
import time

import cv2

import config
from currency import CurrencyClassifier
from detector import Detector
from device import pick_device
from distance import DepthEstimator, PinholeEstimator
from guide import Announcer, Obstacle, direction_of
from speech import Speaker
from tracking import DistanceSmoother

GREEN, RED, WHITE = (0, 200, 0), (0, 0, 255), (255, 255, 255)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(config.CAMERA_INDEX), help="camera index or video file")
    p.add_argument("--depth", action="store_true", help="neural depth instead of pinhole formula")
    p.add_argument("--mute", action="store_true", help="print guidance instead of speaking")
    return p.parse_args()


def draw(frame, obstacles: list[Obstacle], fps: float, mode: str):
    for o in obstacles:
        x1, y1, x2, y2 = o.box
        color = RED if o.distance_m < config.DANGER_DISTANCE_M else GREEN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"#{o.track_id} " if o.track_id is not None else ""
        text += f"{o.label} {o.distance_m:.1f}m"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    w = frame.shape[1]
    for x in (w // 3, 2 * w // 3):  # left | ahead | right zones
        cv2.line(frame, (x, 0), (x, frame.shape[0]), (120, 120, 120), 1)
    cv2.putText(frame, f"FPS {fps:.1f} | {mode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)


def read_currency(classifier, frame, speaker: Speaker):
    if classifier is None:
        speaker.say("Currency model not found. Train it first.")
        return
    label, conf = classifier.predict(frame)
    if conf < config.CURRENCY_MIN_CONFIDENCE or not label.isdigit():
        speaker.say("Not sure. Hold the note closer to the camera.")
    else:
        speaker.say(f"{label} rupees note")


def reconnect(source, speaker: Speaker, timeout_s: float):
    """The camera stopped giving frames. Tell the user at once, then keep trying to reopen it.

    A blind user can't see a frozen window, so silence would sound like a clear path.
    """
    speaker.say("Camera lost. Guidance stopped.", urgent=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(1.0)
        cap = cv2.VideoCapture(source)
        if cap.isOpened() and cap.read()[0]:
            speaker.say("Camera back. Guidance resumed.")
            return cap
        cap.release()
    return None


def main():
    args = parse_args()
    speaker = Speaker(enabled=not args.mute)
    detector = Detector(config.YOLO_WEIGHTS, config.CONF_THRESHOLD)
    if args.depth:
        estimator = DepthEstimator(config.DEPTH_MODEL_ID)
    else:
        estimator = PinholeEstimator(config.FOCAL_LENGTH_PX, config.KNOWN_HEIGHTS_M,
                                     config.DANGER_DISTANCE_M, config.CAMERA_HEIGHT_M)
    smoother = DistanceSmoother(config.DISTANCE_SMOOTHING)
    announcer = Announcer(speaker, config.ANNOUNCE_COOLDOWN_S, config.DANGER_DISTANCE_M,
                          config.MAX_ANNOUNCEMENTS, config.APPROACH_SPEED_MPS,
                          config.TIME_TO_CONTACT_S)
    currency = None
    if os.path.exists(config.CURRENCY_MODEL_PATH):
        currency = CurrencyClassifier(config.CURRENCY_MODEL_PATH, pick_device())

    source = int(args.source) if args.source.isdigit() else args.source
    is_video_file = os.path.isfile(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        speaker.say("Camera not found. Drishti cannot start.", urgent=True)
        speaker.wait_until_quiet()
        raise SystemExit(f"could not open video source {args.source!r}")
    mode = "depth model" if args.depth else "pinhole"
    speaker.say("Drishti-AI is ready")
    prev = time.time()
    fps = 0.0
    # Whatever ends guidance, the user must hear it: silence would sound like a clear path.
    stop_message = "Drishti has stopped."

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                if is_video_file:
                    stop_message = "Video finished. Drishti has stopped."
                    break
                cap = reconnect(source, speaker, config.CAMERA_RETRY_S)
                if cap is None:
                    stop_message = "Camera not working. Drishti has stopped."
                    break
                continue
            scale = config.FRAME_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

            detections = detector(frame)
            distances = estimator.estimate(frame, detections)
            now = time.time()
            obstacles = []
            for d, dist in zip(detections, distances):
                if dist is None:
                    continue
                dist, approach = smoother.update(d.track_id, dist, now)
                obstacles.append(Obstacle(d.label, d.conf, d.box, dist,
                                          direction_of(d.box, frame.shape[1]), d.track_id,
                                          approach))
            smoother.forget_old(now)
            announcer.update(obstacles)

            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            draw(frame, obstacles, fps, mode)
            cv2.imshow("Drishti", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                read_currency(currency, frame, speaker)
            elif key == ord("q"):
                cap.release()
                break
    except Exception:
        stop_message = "Drishti error. Guidance stopped."
        raise
    finally:
        speaker.say(stop_message, urgent=True)
        speaker.wait_until_quiet()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
