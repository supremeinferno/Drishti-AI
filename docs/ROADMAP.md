# Roadmap: from prototype to a real assistive product

The plan, in order. Each phase builds on the previous one. Issue numbers (#) refer to [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

```
Phase 0  Prototype                    ✅ done
Phase 1  Safety fixes                 ⏳ next
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

## Phase 1: Safety fixes ⏳
Goal: the app should never be confidently wrong about danger.

- [ ] Distance correct when the box is cut off at the bottom (#1)
- [ ] Calibration at the same resolution as the live app (#2)
- [ ] Never drop obstacles without a known height; fall back to depth or box size (#3)
- [ ] Priority speech: danger interrupts, plus an instant warning beep (#4)
- [ ] Speak an alert when the camera fails, plus a "system alive" heartbeat (#5)
- [ ] Object tracking (ByteTrack) and smoothed distance per object (#6)
- [ ] Classify currency on a clean frame (#8)
- [ ] Unit tests for distance, direction and the announcer logic (#20)

**Done when:** all 🔴 issues 1–6 are closed and covered by tests.

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

- [ ] Time-to-collision from tracked distance, so warnings depend on how fast something approaches
- [ ] Whitelist of relevant obstacles; ignore objects in the user's own hand (#17)
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

- [ ] Distance benchmark (pinhole vs depth, at 1–5 m)
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
