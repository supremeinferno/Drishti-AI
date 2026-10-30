"""Central settings for Drishti. Tweak these instead of editing the code."""

CAMERA_INDEX = 0
FRAME_WIDTH = 640
CAMERA_RETRY_S = 10.0        # keep trying to reopen a lost camera this long before giving up

# Object detection
YOLO_WEIGHTS = "yolov8n.pt"  # swap for yolov8s.pt or your own fine-tuned best.pt
CONF_THRESHOLD = 0.45

# Distance estimation (pinhole camera model): distance = real_height * focal / pixel_height
# Run calibrate.py once for your webcam and paste the printed value here.
FOCAL_LENGTH_PX = 600.0

# Height of the camera above the floor (chest-mounted phone or held laptop) and kept roughly
# level. Used to measure distance from where an object touches the floor when the top of
# its box is cut off by the frame.
CAMERA_HEIGHT_M = 1.3

# Approximate real-world heights in metres for COCO classes we care about.
KNOWN_HEIGHTS_M = {
    "person": 1.70,
    "bicycle": 1.00,
    "car": 1.50,
    "motorcycle": 1.10,
    "bus": 3.00,
    "truck": 3.00,
    "traffic light": 0.90,
    "stop sign": 0.75,
    "bench": 0.80,
    "dog": 0.50,
    "cat": 0.30,
    "backpack": 0.45,
    "suitcase": 0.60,
    "bottle": 0.25,
    "cup": 0.10,
    "chair": 0.90,
    "couch": 0.85,
    "potted plant": 0.50,
    "bed": 0.60,
    "dining table": 0.75,
    "tv": 0.50,
    "laptop": 0.25,
    "cell phone": 0.15,
    "book": 0.22,
    "refrigerator": 1.70,
    # common on Indian streets
    "cow": 1.40,
    "horse": 1.60,
    "sheep": 0.80,
    "elephant": 2.80,
    "fire hydrant": 0.75,
    "parking meter": 1.30,
    "train": 4.00,
}
# Classes missing from this table are still measured, from where they stand on the floor.

# Optional neural depth model (python main.py --depth). Outputs metric depth in metres.
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# Approximate width in metres, used as a second estimate when the height can't be trusted:
# a seated person or one cut off by the frame still shows their full shoulders.
KNOWN_WIDTHS_M = {
    "person": 0.45,
}

# What is worth saying out loud. Everything else is still drawn on screen, in grey, but never
# spoken: a cup, a book or the phone in the user's own hand is not an obstacle.
ANNOUNCE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "cat", "dog", "horse", "sheep", "cow", "elephant",
    "fire hydrant", "parking meter", "bench", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "refrigerator", "suitcase",
}
ANNOUNCE_MIN_CONF = 0.55     # stricter than CONF_THRESHOLD: track at 0.45, only speak when surer
CONFIRM_S = 0.5              # an object must be tracked this long before it is spoken
CONFIRM_DANGER_S = 0.2       # ...but something already close can't wait that long

# Voice guidance
DANGER_DISTANCE_M = 1.0      # closer than this -> "Careful!" and red box
ANNOUNCE_COOLDOWN_S = 10.0   # an object that is still there is repeated after this many seconds
MAX_ANNOUNCEMENTS = 2        # only speak about the N nearest obstacles

# Tracking: the same object keeps one ID across frames, so its distance can be averaged
# and we can tell when something is coming towards the user.
DISTANCE_SMOOTHING = 0.4     # weight of the newest frame (lower = steadier, but slower)
APPROACH_SPEED_MPS = 1.5     # faster than walking pace -> the object itself is moving at us
TIME_TO_CONTACT_S = 3.0      # warn when it would reach the user within this many seconds

# Currency recognition
CURRENCY_MODEL_PATH = "models/currency.pt"
CURRENCY_MIN_CONFIDENCE = 0.60
