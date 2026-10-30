# Roadmap: from prototype to a real assistive product

The plan, in order. Each phase builds on the previous one. Issue numbers (#) refer to [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

```
Phase 0  Prototype                    ✅ done
Phase 1  Safety fixes                 ✅ mostly done (tests still to write)
Phase 1b Real-world accuracy          ⏳ now
Phase 2  Indian hazard detection
Phase 3  Currency v2
Phase 4  Smarter guidance
Phase 5  Mobile / wearable, offline, multilingual
Phase 6  Evaluation and user study    (runs alongside every phase)
Phase 7  Extra features
```

---

## Phase 0: Prototype ✅
- YOLOv8 detection, pinhole and neural depth, left / ahead / right directions, voice with cooldown.
- Currency data collection and training pipeline with three architectures.

## Phase 1: Safety fixes ✅ (mostly)
Goal: the app should never be confidently wrong about danger.

- [x] Distance correct when the box is cut off by the frame (#1)
- [x] Calibration at the same resolution as the live app (#2)
- [x] Never drop obstacles without a known height (#3)
- [x] Priority speech: danger interrupts, plus an instant warning beep (#4)
- [x] Speak an alert when the camera fails; announce every way of stopping (#5)
- [x] Object tracking (ByteTrack), smoothed distance, approach warnings, "still there" repeats (#6)
- [x] Speak only real, confirmed obstacles (#17, part)
- [x] Shoulder-width distance for seated or cut-off people (#1)
- [ ] Classify currency on a clean frame (#8)
- [ ] Unit tests for distance, tracking, the announce filter, the announcer and speech priority (#20)
- [ ] Optional "system alive" heartbeat tone

## Phase 1b: Real-world accuracy ⏳
Goal: what the user hears matches reality in a real walking setup.

- [ ] Calibrate focal length and camera height on each team laptop and phone (#14)
- [ ] Record a ~1 min chest-height walking video and use it as a fixed test for every change
- [ ] Screen and speech match: on-screen subtitle of the last spoken sentence, rounded spoken distance, re-announce on large changes (#22)
- [ ] Floor-based distance only when the camera setup fits; camera-tilt setting (#21)
- [ ] Compare `yolov8n` / `yolo11n` / `yolo11s` on accuracy and FPS (#23)
- [ ] Test the speech interrupt on Windows (#24)

## Phase 2: Indian hazard detection
Goal: detect what actually hurts blind pedestrians in India.

- [ ] Define classes: stairs (up / down), pothole, open drain, speed breaker, pole, wall, glass door, auto-rickshaw, e-rickshaw, two-wheeler, cow / dog, footpath edge
- [ ] Collect our own videos (phone at chest height on local streets) and add public datasets
- [ ] Label in Roboflow or CVAT; train / val / test split by **location**
- [ ] Fine-tune YOLO (compare v8n / YOLO11n / YOLO11s)
- [ ] **Free-path detection:** use the depth map or ground segmentation to warn about *any* obstacle in the walking corridor, even unknown ones (#7)
- [ ] Head-height hazards (branches, mirrors, signboards) that a cane misses

**Done when:** we report mAP per class on a held-out test set from new locations.

## Phase 3: Currency v2
Goal: reliable, hands-free note reading in a real shop.

- [ ] Re-collect the data: several sessions, places, lights and hands per class; old and new designs (#11, #13)
- [ ] Split by session; separate untouched test set (#9, #10)
- [ ] Find the note first, then classify the crop (#12)
- [ ] Vote over several frames before speaking (#12)
- [ ] Hands-free trigger: voice command or automatic when a note fills the frame (#16)
- [ ] Count several notes ("two 500s and one 100, total 1100 rupees")
- [ ] Grad-CAM check that the model looks at the note
- [ ] Export the best model to ONNX / TFLite for mobile

**Done when:** test accuracy and per-class F1 are reported on a test set collected on a different day and camera.

## Phase 4: Smarter guidance
Goal: say less, but say exactly the right thing.

- [x] Time-to-collision from tracked distance (approach warnings)
- [x] Whitelist of relevant obstacles (#17)
- [ ] Ignore objects held in the user's own hand, even when they are obstacle classes
- [ ] Distances in steps; optional clock directions (#17)
- [ ] Path advice: "Obstacle ahead, move slightly right" (using the free-path map from Phase 2)
- [ ] Verbosity modes: *quiet* (danger only), *normal*, *describe scene*

## Phase 5: Mobile / wearable, offline, multilingual
Goal: something a blind person can carry and use daily.

- [ ] `--headless` mode with no display window (#19)
- [ ] Hindi plus at least one regional language voice (#18)
- [ ] Optimise: INT8 quantisation, ONNX / TFLite / CoreML, depth on alternate frames (#15)
- [ ] Target A: **Android app** (camera + on-device models + TTS + vibration)
- [ ] Target B: **wearable**: Raspberry Pi 5 or Jetson Orin Nano, chest camera, bone-conduction earphones, a button
- [ ] Battery and heat test: at least 2 hours of continuous use
- [ ] Fully offline: no internet needed

## Phase 6: Evaluation and user study (alongside every phase)
See [EVALUATION.md](EVALUATION.md).

- [ ] Distance benchmark (pinhole vs depth, at 1–5 m, standing and seated)
- [ ] Detection mAP on our own Indian test set
- [ ] Currency test accuracy and confusion matrix
- [ ] Speed (FPS, latency) on the laptop and the target device
- [ ] Controlled walking course: collisions and time, with and without Drishti
- [ ] Feedback sessions with visually impaired volunteers (partner with an NGO or a school for the blind), with informed consent

## Phase 7: Extra features
- OCR for signboards, bus numbers and medicine labels (EasyOCR / PaddleOCR)
- Scene description on demand ("What's in front of me?") using a small vision-language model
- Face recognition of known people (opt-in, privacy-first)
- Emergency SOS with location
- Navigation integration (GPS turn-by-turn plus obstacle avoidance)
