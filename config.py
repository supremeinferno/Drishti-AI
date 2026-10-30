"""Central settings for Drishti. Tweak these instead of editing the code."""

CAMERA_INDEX = 0
FRAME_WIDTH = 640

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
}

# Optional neural depth model (python main.py --depth). Outputs metric depth in metres.
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# Voice guidance
DANGER_DISTANCE_M = 1.0      # closer than this -> "Careful!" and red box
ANNOUNCE_COOLDOWN_S = 4.0    # don't repeat the same object+direction faster than this
MAX_ANNOUNCEMENTS = 2        # only speak about the N nearest obstacles

# Tracking: the same object keeps one ID across frames, so its distance can be averaged
# and we can tell when something is coming towards the user.
DISTANCE_SMOOTHING = 0.4     # weight of the newest frame (lower = steadier, but slower)
APPROACH_SPEED_MPS = 1.5     # faster than walking pace -> the object itself is moving at us
TIME_TO_CONTACT_S = 3.0      # warn when it would reach the user within this many seconds

# Currency recognition
CURRENCY_MODEL_PATH = "models/currency.pt"
CURRENCY_MIN_CONFIDENCE = 0.60
