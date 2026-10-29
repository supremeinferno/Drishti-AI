# Third Eye – AI Assistant for the Visually Impaired

A real-time assistant that tells a blind or low-vision user what is around them, how far away it is and in which direction, and reads Indian currency notes out loud.

> "Careful! chair, 0.8 meters ahead. person, 2.4 meters on your left."

## Pipeline

```
Webcam frame
   ├─► YOLOv8 object detection ──► bounding boxes + labels
   │        └─► Distance estimation
   │              • Pinhole model: d = H_real × f / h_pixels   (fast, explainable)
   │              • Depth Anything V2 metric depth (--depth)   (neural, any object)
   │        └─► Direction: left / ahead / right (frame split into thirds)
   │        └─► Announcer: nearest N obstacles, cooldown, "Careful!" under 1 m
   │                 └─► Text-to-speech (background thread)
   └─► [C key] Currency classifier (MobileNetV3 / EfficientNet-B0 / ResNet50,
                transfer learning on our own dataset) ──► "500 rupees note"
```

## Setup

Use Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
# NVIDIA laptop: install the CUDA build of torch first, from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Steps

1. **Run the obstacle detector:** `python main.py` (the YOLO weights download on first run).
2. **Calibrate distance:** stand at a measured 2 m from the camera, run `python calibrate.py --distance 2.0 --height 1.70`, press SPACE 5–10 times, then Q. Paste the printed `FOCAL_LENGTH_PX` into `config.py`.
3. **Collect a currency dataset**, about 150–300 photos per class:
   ```bash
   python collect_currency.py --label 10
   python collect_currency.py --label 20
   ...                         # 50, 100, 200, 500
   python collect_currency.py --label background   # no note in view
   ```
   Vary the angle, distance, lighting and background, and include folded or partly covered notes.
4. **Split the data:** `python split_dataset.py`
5. **Train the classifier on the laptop GPU:**
   ```bash
   python train_currency.py --arch mobilenet_v3_large
   python train_currency.py --arch efficientnet_b0 --out models/currency_effb0.pt
   python train_currency.py --arch resnet50 --batch 8 --out models/currency_r50.pt
   ```
   Mixed precision is turned on automatically on CUDA. If you run out of memory, lower `--batch`. The script prints a classification report and confusion matrix for the report.
6. **Full demo:** `python main.py`, then hold a note in front of the camera and press **C**.
   Try `python main.py --depth` to compare against the neural depth model.

## For the report

- **Model comparison table:** accuracy, F1, model size and inference FPS for the three architectures.
- **Distance accuracy:** place objects at 1, 2, 3 and 4 m, then record the estimated distance and the error for pinhole vs depth model.
- **Real-world test:** walk a corridor with the laptop and record the run.
- **Limitations:** pinhole needs known object heights; depth model is slower; low light.

## Extensions

- Fine-tune YOLO on custom classes such as stairs, potholes, open doors and Indian traffic signs (label them in Roboflow, then run `yolo train`).
- OCR mode for reading signboards and medicine labels (EasyOCR).
- Run on a phone camera through an IP Webcam stream: `python main.py --source http://<phone-ip>:8080/video`.
