# Drishti AI: Progress Check ki Taiyari 📚

> Yeh aapke padhne ke liye hai. Har section mein 4 cheezein hain: **kya use kiya**, **uska kaam kya hai**, **wahi kyun liya**, aur **kuch aur kyun nahi liya**. End mein **results**, **ma'am ke possible sawaal**, aur **demo ka plan** hai.

---

## 0. Ek line mein project (sabse pehle yeh bolna)

> "Drishti AI ek real-time assistant hai jo camera se dekh kar blind insaan ko **bolke** batata hai ki saamne **kya** hai, **kitni door** hai, aur **kis taraf** hai. Agar koi cheez paas aa rahi ho to beep ke saath turant warning deta hai. Saath mein Indian currency notes pehchaanne ka module bhi hai."

**Problem kyun important hai:** white cane sirf ~1 m tak aur zameen ke level par kaam karta hai. Sir ki height wali cheezein (shisha, branch, board) aur door ki cheezein nahi pakad pata. Indian roads par gaay, auto, gaddhe, khule naale hain. Seeing AI aur Lookout jaise apps Indian conditions ke liye nahi bane, aur unhe aksar internet chahiye.

---

## 1. Poora flow: ek frame ke saath kya hota hai

```
Camera → frame (640 px chauda)
  1. YOLOv8 + ByteTrack  → "person", 91%, box, ID #209
  2. Distance            → 0.84 m
  3. Smoothing           → distance stable, speed = paas aa raha hai ya nahi
  4. Direction           → left / ahead / right
  5. Filter              → bolne layak hai? (asli rukawat? pakka? 0.5 s se dikh raha?)
  6. Announcer           → kya bolna hai, kab, urgent hai ya nahi
  7. Speaker             → 🔔 beep + "Careful! person, 0.8 meters ahead"
```

Yeh poora loop **~20 baar per second** chalta hai (aapke Mac par FPS 20.1 aaya tha).

**Har file ka kaam (yaad kar lo):**

| File | Kaam |
|---|---|
| `main.py` | Camera loop, sab modules jodna, screen par drawing, camera lost handling |
| `config.py` | Saari settings ek jagah (thresholds, heights, kaun si cheezein bolni hain) |
| `detector.py` | YOLO + ByteTrack: object pehchaanna aur ID dena |
| `distance.py` | Distance nikalna (pinhole formula ya depth model) |
| `tracking.py` | Har object ki distance smooth karna, speed nikalna |
| `guide.py` | Kya bolna hai, kab bolna hai, kya urgent hai |
| `speech.py` | Background mein bolna, urgent ko priority |
| `device.py` | GPU chunna (NVIDIA → cuda, Mac → mps, warna cpu) |
| `currency.py`, `train_currency.py`, `collect_currency.py`, `split_dataset.py` | Currency module: data collect, split, train, predict |
| `calibrate.py` | Camera ka focal length naapna |

**Design ka idea:** har kaam alag file mein hai (**modular**). Isliye ek cheez badalni ho, jaise naya detector ya naya distance method, to baaki code nahi chhedna padta. Settings `config.py` mein hain, code mein hardcoded nahi.

---

## 2. Object Detection: YOLOv8n

### Kya hai
YOLO = **"You Only Look Once"**. Yeh ek neural network hai jo **ek hi baar** image dekh kar saare objects ke **box + label + confidence** de deta hai. Hamara version **YOLOv8n** hai (n = nano, sabse chhota), Ultralytics library se.

- **COCO dataset** par pre-trained: **80 classes** (person, car, chair, dog, cow…), ~1.18 lakh training images.
- Size: **~3.2 million parameters**. COCO par accuracy **mAP 37.3**.

### Kaam
Har frame mein batana ki kya hai aur kahan hai: `person 91% box(186, 59, 507, 356)`.

### Yahi kyun
1. **Real-time:** one-stage detector hai, ek pass mein sab kar deta hai. Laptop par ~20 FPS.
2. **Nano model:** chhota hai, to aage chal kar phone ya Raspberry Pi par bhi chal sakta hai.
3. **Pre-trained:** 80 classes free mein mil jaati hain. Shuru mein apna data label nahi karna pada.
4. **Ultralytics library:** tracking, training aur export (phone ke liye) sab built-in hai.

### Kuch aur kyun nahi
| Alternative | Kyun nahi liya |
|---|---|
| **Faster R-CNN** | Two-stage detector hai: pehle regions dhoondhta hai, phir classify karta hai. Accurate hai lekin **bahut slow** (laptop par kuch hi FPS). Real-time guidance ke liye bekaar |
| **SSD / MobileNet-SSD** | Fast hai lekin purana hai. Same speed par YOLO zyada accurate hai |
| **DETR (transformer)** | Heavy aur slow, phone par mushkil |
| **YOLOv8s / m / l** | Zyada accurate, lekin slow. Pehle chhote se shuru kiya. Ab compare karne ka plan hai (YOLO11n mAP 39.5, YOLO11s mAP 47.0) |

### Important terms (ma'am pooch sakti hain)
- **Confidence:** model kitna sure hai (0–1). Hum **0.45** se upar wale detect karte hain, aur bolte sirf **0.55** se upar wale hain.
- **Bounding box:** `(x1, y1, x2, y2)`, object ke chaaron taraf rectangle.
- **IoU (Intersection over Union):** do boxes kitna overlap karte hain = overlap area ÷ dono ka combined area.
- **mAP (mean Average Precision):** detector ki accuracy ka standard number. Har class ka precision-recall curve ka area nikalo, phir sab classes ka average lo.

### Limitation (khud bolna, achha impression padta hai)
- COCO mein **stairs, gaddhe, naale, khambe nahi hain**, isliye inhe detect hi nahi kar sakta. Plan hai apna Indian dataset bana kar fine-tune karna (Phase 2).
- Chhota model kabhi galat label deta hai. Testing mein **purse ko "book" aur "laptop"** bola. Isliye bade model se comparison plan mein hai.

---

## 3. Object Tracking: ByteTrack (Sachin ne add kiya)

### Kya hai
Har object ko ek **ID** deta hai jo frames ke beech same rehti hai: kursi #12 agle frame mein bhi #12 rahegi.

### Kaam
Bina tracking ke har frame alag hota hai. Isse:
- distance har frame ups-and-down hoti (noise),
- pata nahi chalta ki koi cheez **paas aa rahi hai**,
- ek hi kursi baar-baar announce hoti.

### Kaise kaam karta hai (simple mein)
1. Har object ka agla position predict karta hai (**Kalman filter**, motion model).
2. Naye boxes ko purane tracks se IoU ke basis par match karta hai.
3. **ByteTrack ki khaas baat:** low-confidence boxes ko bhi second round mein match karta hai. To agar object thoda chhup gaya aur confidence gir gaya, tab bhi ID nahi tootti.

### Kuch aur kyun nahi
| Alternative | Kyun nahi |
|---|---|
| **SORT** | Simple hai, lekin low-confidence boxes phenk deta hai, isliye IDs zyada toot-ti hain |
| **DeepSORT** | Ek extra neural network (re-ID) chalata hai appearance ke liye. **Slow** hai aur hamare use ke liye zaroori nahi |
| **Khud ka tracker** | Ultralytics mein ByteTrack built-in hai aur tested hai. Dobara banana bekaar |

---

## 4. Distance Estimation: project ka sabse important hissa

### Method 1: Pinhole Camera Model (default)

**Formula:**
```
distance = real_height × focal_length ÷ box_height_in_pixels
```

**Kyun kaam karta hai:** cheez jitni door hogi, image mein utni chhoti dikhegi (similar triangles). Agar asli height pata hai aur image mein kitne pixel ki hai yeh pata hai, to distance nikal sakte hain.

**Example:** person 1.7 m, focal 600 px, box 340 px tall → 1.7 × 600 ÷ 340 = **3.0 m**

**Focal length kya hai:** camera ki ek property, pixels mein. Har camera ki alag hoti hai, isliye `calibrate.py` se naapte hain: 2 m par khade ho, box ki height naapo, aur formula ulta karo: `focal = pixel_height × distance ÷ real_height`. 5–10 samples ka **median** lete hain (average nahi, kyunki median outliers se kam affect hota hai).

**Known heights:** `config.py` mein person 1.70 m, chair 0.90 m, car 1.50 m, cow 1.40 m… total 32 classes.

### Is formula ki problems, aur humne kaise solve kiya ⭐ (yeh achhe se samjhao, yahi asli kaam hai)

**Problem 1: box frame se kat gaya.** Paas khade insaan ke pair frame ke neeche chale jaate hain. Box chhota ho jaata hai, to formula **door** batata hai, jabki woh paas hai. **Khatarnak!**
→ **Solution (Sachin + humne):** jo edge dikh raha hai usse naapo, **ground-plane formula** se:
```
distance = (camera_height − point_height) × focal ÷ (pixels below image centre)
```
Camera zameen se 1.3 m par hai. Zameen par jo point image mein jitna neeche dikhe, utna paas hai.
→ **Bug jo humne pakda:** "kata hua" check sirf 2 pixel ka tha, lekin YOLO edge se thoda pehle ruk jaata hai, to check kabhi chala hi nahi. Ab **frame ka 2%** margin hai.

**Problem 2: baitha hua insaan.** Height 1.7 m maan lete hain, lekin baitha insaan chhota dikhta hai, to distance galat aati hai.
→ **Solution:** **kandhe ki chaudai (0.45 m)** se bhi distance nikalo: `0.45 × focal ÷ box_width`. Baithne se chaudai nahi badalti.

**Problem 3: dono estimates mein se kaunsa lein?** → Hamesha **chhota (paas wala)**.
> **Ma'am ko yeh line zaroor bolna:** "Agar galti karni hi hai, to cheez ko paas batana safe hai. Door batana user ko chot pahuncha sakta hai. Isliye hum hamesha conservative estimate lete hain."

**Problem 4: jinki height pata nahi.** Pehle aise objects **chupchaap drop** ho jaate the: gaay detect hui, lekin bola kuch nahi.
→ **Solution:** jahan object zameen ko touch karta hai, wahan se naapo (ground-plane). Agar object table par hai (horizon se upar), to woh raaste mein nahi hai, isliye nahi bolte.

**Result (aapke screenshot se):** aap ~0.6 m par baithe the.
| Version | Distance |
|---|---|
| Purana code | **3.4 m** ❌ |
| Edge bug fix ke baad | 1.98 m |
| Shoulder width ke baad | **0.84 m** |
| Calibration ke baad (andaza) | ~0.64 m |

### Method 2: Depth Anything V2 (`--depth` flag)
- Ek neural network (2024) jo **har pixel ki depth meters mein** batata hai, ek hi normal camera se (**monocular depth**). Hum "Small" version use karte hain (~25M params).
- Box ke **beech ke 50% hisse ka median** lete hain, taaki background ke pixels result na bigaadein.
- **Faayda:** kisi bhi object ki distance, height jaane bina. **Nuksaan:** bahut slow, aur "Indoor" model hai, outdoor par kam accurate.

### Dono kyun rakhe
Comparison ke liye. Pinhole **fast aur explainable** hai (formula samjha sakte ho). Depth **flexible** hai lekin slow aur "black box". Report mein dono ka error compare karenge.

### Kuch aur kyun nahi
| Alternative | Kyun nahi |
|---|---|
| **Stereo camera (2 cameras)** | Accurate hai, lekin extra hardware chahiye aur calibration mushkil hai. Hum sirf phone ya laptop ke ek camera se karna chahte hain |
| **LiDAR / ultrasonic sensor** | Mehenga aur extra hardware. Har blind user ke paas nahi hoga. Smart canes isi wajah se costly hain |
| **Sirf depth model** | Slow hai, real-time mein lag aata hai |

---

## 5. Smoothing aur Approach Speed (`tracking.py`)

**Smoothing: EMA (Exponential Moving Average)**
```
smoothed = 0.4 × naya_reading + 0.6 × purana_smoothed
```
Ek frame ki reading tens of cm se jumping karti hai. EMA purani values ko yaad rakh kar number stable karta hai. 0.4 = naye frame ka weight: kam rakhoge to zyada stable lekin slow react.

**Kyun EMA, simple average kyun nahi:** EMA ko sirf pichli value yaad rakhni padti hai (fast, kam memory), aur naye data ko zyada weight deta hai.

**Approach speed:** `(purani distance − nayi distance) ÷ time`. Agar speed **≥ 1.5 m/s** (chalne se tez, matlab object khud aa raha hai) **aur** 3 sec mein takkar ho jaayegi (**time-to-contact = distance ÷ speed < 3 s**), to *"Careful! car approaching on your right"*.

---

## 6. Kya bolna hai: Filter + Announcer (`guide.py`)

### Direction
Frame ko 3 hisson mein baanta hai. Box ka centre left third mein ho to "on your left", beech mein "ahead", right mein "on your right".

### AnnounceFilter: bolne se pehle 3 checks
1. **Asli rukawat hai?** `ANNOUNCE_CLASSES` list: person, vehicles, animals, furniture. Phone, cup, book, laptop **nahi** bolte. (Pehle haath ka phone bhi *"Careful! cell phone"* bolta tha!)
2. **Pakka hai?** Confidence ≥ 55%.
3. **0.5 sec se dikh raha hai?** 1 frame ka "ghost" detection nahi bolte. Lekin agar cheez **1 m se paas** hai to sirf 0.2 sec wait karte hain, kyunki khatre mein der nahi kar sakte.

Jo nahi bolte, woh screen par **grey** box mein dikhte hain, taaki debugging ho sake.

### Announcer: kab bolna hai
- Paas aa rahe objects **pehle**, phir sabse paas wale. Ek baar mein **max 2** objects, taaki user overload na ho.
- Naya object → turant bolo. Wahi object abhi bhi hai → **10 sec** baad *"chair still ahead"* (Sachin ka cooldown).
- Jaldi dobara bolo agar: danger zone mein 0.5 m aur paas aa gaya, ya approach karna shuru kiya.
- **1 m se paas = "Careful!"** + red box + urgent.

**Kyun itne rules:** blind user ke liye zyada bolna bhi khatarnak hai. Woh traffic ki awaaz, logon ki awaaz nahi sun paayega. Isliye "sirf zaroori cheez, sahi waqt par".

---

## 7. Speech: Priority System (`speech.py`)

### Kaise kaam karta hai
- **Alag thread** mein bolta hai. Agar main loop mein bolta to bolte waqt video ruk jaata (blocking).
- **Do slots: urgent aur normal.** Har slot mein sirf **sabse naya** message rehta hai. Purane messages pile up nahi hote, kyunki 5 sec purani info bekaar hai.
- **Urgent message:**
  - normal ko **beech mein kaat deta hai**,
  - normal message use **kabhi replace nahi kar sakta**,
  - **turant beep** bajata hai, voice shuru hone se pehle.
- macOS par built-in `say` command, Windows/Linux par `pyttsx3`.

**Pehle bug kya tha:** queue mein sirf 1 message tha. *"Careful! chair"* ko peeche se *"person 3 m"* mita deta tha, aur warning ~2 sec tak current sentence khatam hone ka wait karti thi.

### Kuch aur kyun nahi
| Alternative | Kyun nahi |
|---|---|
| **Google TTS / cloud voices** | Internet chahiye. Blind user ka net chala gaya to app band. Hum **offline** chahte hain |
| **pyttsx3 Mac par** | Background thread mein Mac par unreliable hai (crash/hang). `say` stable hai aur process ko kill karke turant rok sakte hain |
| **Simple queue (FIFO)** | Purane messages line mein lagte, user ko stale info milti |

### Camera Safety (`main.py`)
Camera band ho jaaye to: *"Camera lost. Guidance stopped."*, phir 10 sec tak reconnect ki koshish → *"Camera back."* App kisi bhi wajah se band ho (Q, crash, video khatam), **hamesha bolta hai**.
> **Line:** "Blind user ke liye silence ka matlab hota hai ki raasta saaf hai. Isliye app ka chupchaap band hona sabse khatarnak bug tha."

---

## 8. Currency Module

### Pipeline
```
collect_currency.py → webcam se photos (har note ki 150–300)
split_dataset.py    → 80% train, 20% validation
train_currency.py   → model train
main.py (C key)     → "500 rupees note"
```
**Classes:** 10, 20, 50, 100, 200, 500, **background** ("background" class isliye hai taaki model bol sake "koi note nahi hai", warna woh kuch bhi zabardasti guess karta).

### Transfer Learning: kya aur kyun
ImageNet (~13 lakh images, 1000 classes) par trained model lete hain. Woh pehle se edges, textures, shapes pehchaanta hai. Sirf **last layer** badal kar 7 classes ke liye train karte hain.
**Kyun:** hamare paas sirf ~1500–2000 photos hongi. Zero se train karne ke liye lakhon chahiye hoti. Transfer learning chhote data par bhi achha kaam karta hai aur jaldi train hota hai.

### 3 Models compare kyun
| Model | Params | ImageNet accuracy | Kyun |
|---|---|---|---|
| **MobileNetV3-Large** | ~5.5M | ~75% | Phone ke liye bana hai, fast. Hamara main choice |
| **EfficientNet-B0** | ~5.3M | ~77.7% | Speed aur accuracy ka balance |
| **ResNet50** | ~25.6M | ~80.9% | Strong baseline, bada aur slow |

Comparison se dikhega ki chhota model kitni accuracy deta hai, aur phone ke liye sahi trade-off kya hai.

### Training choices (har ek ka reason)
| Choice | Kya | Kyun |
|---|---|---|
| **Augmentation** | Random crop, ±20° rotation, brightness/contrast change, perspective | Asli duniya mein note tedha, door, kam roshni mein hoga. Model ko yeh sab pehle hi dikha do |
| **Horizontal flip NAHI** | — | Mirror kiya hua note asli duniya mein kabhi nahi dikhta, to woh model ko confuse karega |
| **AdamW optimizer** | lr 3e-4 | Adam + sahi weight decay. Transfer learning ke liye reliable default |
| **Cosine LR schedule** | Learning rate dheere-dheere kam | Shuru mein bade steps, end mein chhote. Better convergence |
| **Label smoothing 0.1** | 100% ki jagah 90% sure hona sikhaate hain | Model over-confident nahi hota, aur confidence threshold (0.60) zyada meaningful hota hai |
| **Mixed precision (AMP)** | NVIDIA par float16 | ~2× fast training, kam GPU memory |
| **ImageNet mean/std normalization** | — | Pretrained model isi format mein trained tha |
| **Confidence < 0.60 → "Not sure"** | — | Galat note batana (₹100 ko ₹500) paise ka nuksaan hai. "Not sure" bolna safe hai |

### Metrics jo script print karti hai
- **Accuracy:** kitne % sahi.
- **Precision:** jab model ne "500" bola, kitni baar sach mein 500 tha.
- **Recall:** asli 500 notes mein se kitne pakde.
- **F1:** precision aur recall ka harmonic mean.
- **Confusion matrix:** kaunsa note kis ke saath confuse hota hai (jaise ₹10 vs ₹20).

### ⚠️ Status: sach batana
**Model abhi train nahi hua.** `data/` aur `models/` folders abhi nahi hain. Code poora ready hai, dataset collect karna baaki hai. Aur humne already pehchaan liya hai ki dhyan kya rakhna hai:
- Burst mode mein lagbhag same photos train aur val dono mein chali jaayengi, aur accuracy jhoothi badi aayegi. → Session-wise split karenge.
- Har note alag din, alag background, alag roshni mein.
- Alag **test set** jo sirf ek baar use ho.

---

## 9. Results: abhi kya hai aur kya nahi (imaandari se)

**✅ Jo kaam karta hai aur dikhaya ja sakta hai:**
- Laptop par real-time **~20 FPS**, detection, tracking aur voice.
- Seated person distance **3.4 m → 0.84 m** (galat → sahi ke kareeb), screenshot ke saath.
- Haath ka phone aur galat-label purse ab nahi bolte (grey box).
- Priority speech: test mein urgent ne normal ko 0.5 sec par kaata, aur waiting urgent ko normal replace nahi kar paaya.
- Camera lost, video khatam, missing camera: teeno test mein sahi bole.

**❌ Abhi nahi hai (aur plan kya hai):**
| Kya nahi hai | Plan |
|---|---|
| Distance ke proper numbers (1–5 m par error) | `docs/EVALUATION.md` mein test plan ready hai. Calibration ke baad karenge |
| Currency accuracy | Dataset collect → train → test set par report |
| Stairs / gaddhe detection | Apna dataset + YOLO fine-tune |
| Blind users ke saath testing | NGO / blind school se, consent ke saath |

> **Line:** "Humne results ke tables pehle se design kar liye hain (`docs/EVALUATION.md`), aur unhe sirf real measurement se bharenge, andaze se nahi."

---

## 10. Project management (yeh bhi marks dilata hai)
- **GitHub** par code, **Pull Requests** se teammates ka kaam merge (PR #1 tracking, PR #2 cooldown, Sachin ke).
- **Documentation:** `README.md` + `docs/` mein 4 files: ARCHITECTURE (kaise kaam karta hai), KNOWN_ISSUES (24 issues, fixed aur open), ROADMAP (phases), EVALUATION (testing plan).
- **Issue tracking:** har bug ka number, severity (🔴 safety / 🟠 correctness / 🟡 usability), file, wajah aur fix likha hai.
- **Safety-first priority:** pehle woh bugs fix kiye jo user ko chot pahuncha sakte the.

---

## 11. Ma'am ke possible sawaal aur jawab

**Q: YOLO hi kyun?**
Real-time chahiye tha. One-stage detector hai, laptop par 20 FPS. Nano version phone par bhi chal sakta hai. Faster R-CNN zyada accurate hai lekin bahut slow.

**Q: Ek camera se distance kaise nikalte ho?**
Pinhole model: door ki cheez chhoti dikhti hai. `distance = real height × focal ÷ pixel height`. Focal length calibration se naapi. Saath mein ground-plane method aur kandhe ki chaudai bhi use karte hain.

**Q: Distance kitna accurate hai?**
Abhi formal benchmark nahi kiya. Ek test mein seated person ka error 3.4 m se 0.84 m par aaya (asli ~0.6 m). Calibration ke baad 1–5 m par proper test plan hai.

**Q: Agar object ki height nahi pata?**
Jahan woh zameen ko touch karta hai, wahan se camera height use karke naapte hain. Ya `--depth` mode mein neural network se.

**Q: Bahut saari cheezein ek saath ho to?**
Sirf asli rukawatein, sirf pakki (55%+, 0.5 s), max 2 ek baar mein, paas aane wali pehle. Ek hi object 10 sec se pehle repeat nahi hota, jab tak khatra na badhe.

**Q: Tracking kyun chahiye?**
Distance smooth karne, approach speed nikalne, aur ek hi object ko baar-baar na bolne ke liye.

**Q: Internet chahiye?**
Nahi, sab offline chalta hai. Speech bhi offline hai.

**Q: Transfer learning kya hai?**
ImageNet par trained model lekar sirf last layer apne 7 classes ke liye train karna. Chhote dataset par achha kaam karta hai.

**Q: Horizontal flip kyun nahi kiya?**
Mirror note asli mein nahi hota. Yeh model ko galat cheez sikhayega.

**Q: Overfitting kaise rokoge?**
Augmentation, label smoothing, weight decay, pretrained weights, aur best validation epoch save karna. Saath mein session-wise split taaki leakage na ho.

**Q: Isme naya kya hai? Aisa project to bahut logon ne banaya hai.**
(1) **Safety-first design:** conservative distance, priority speech, camera-loss alerts, ghost-detection filter. (2) **Indian context:** cattle, Indian currency, aage chal kar Indian hazards dataset aur Hindi voice. (3) **Do distance methods ka comparison.** (4) **Imaandaar evaluation plan** jisme leakage aur bias pehle se pehchaane gaye hain.

**Q: Limitations kya hain?**
Stairs/gaddhe detect nahi hote (COCO mein nahi hain). Chhota model kabhi galat label deta hai. Distance ke liye calibration chahiye. Floor method maanta hai ki camera seedha chest height par hai. Abhi sirf English. Abhi laptop par chalta hai.

**Q: Aage kya karoge?**
Calibration aur measured results → screen-speech match → bade model se comparison → Indian hazard dataset → currency train → Hindi voice → phone app ya wearable → blind users ke saath testing.

**Q: Kisne kya kiya?**
Git history saaf dikhati hai: initial pipeline, safety fixes aur docs (Pranav), tracking aur cooldown (Sachin, PR #1 aur #2). Apna hissa aap khud sahi-sahi batana. Agar AI tools (jaise Claude) se help li hai aur ma'am poochein, to **sach bolna**. Asli cheez yeh hai ki aap har line samjha sako, aur yeh document isi liye hai.

---

## 12. Kal ke demo ka plan ✅

**Aaj raat zaroor karo:**
1. **Calibration** (5 min). Bina iske distance galat dikhegi:
   ```bash
   python calibrate.py --distance 2.0 --height <aapki height meters mein>
   ```
   SPACE 5–10 baar, phir Q. Jo `FOCAL_LENGTH_PX` print ho, woh `config.py` line 13 mein daalo.
2. `python main.py` ek baar chala kar check karo ki sab theek chal raha hai.
3. Backup ke liye ek **chhota video** record karo. Demo ke time camera fail ho jaaye to `python main.py --source video.mp4`.

**Demo order (~5 min):**
1. Problem (30 sec) → pipeline diagram (README) dikhao.
2. `python main.py` → kisi ko 2–3 m par khada karo → *"person, 2.5 meters ahead"*.
3. Woh paas aaye → 🔔 beep + *"Careful! person…"*, box red.
4. Haath mein phone pakdo → **grey box, kuch nahi bolta**. Explain karo kyun.
5. Camera ke saamne haath rakho ya Q dabao → *"Drishti has stopped"*. Explain karo ki silence kyun khatarnak hai.
6. `docs/KNOWN_ISSUES.md` kholo → fixed vs open issues dikhao.
7. `docs/ROADMAP.md` → aage ka plan.

**Demo mein currency mat dikhana.** Model trained nahi hai. Agar poochein to code aur training pipeline dikhao, aur batao ki dataset collect ho raha hai.

---

## 13. Ek page revision (exam se 5 min pehle)

- **YOLOv8n:** 80 classes, real-time, ~3.2M params, mAP 37.3. Faster R-CNN se fast.
- **ByteTrack:** IDs across frames, low-confidence boxes bhi match karta hai. DeepSORT se fast.
- **Pinhole:** `d = H × f ÷ h`. Calibration se f. Kata box → ground-plane. Baitha → kandhe ki width. **Hamesha chhota estimate (safe).**
- **Depth Anything V2:** per-pixel metric depth, slow, indoor.
- **EMA smoothing 0.4**, approach ≥ 1.5 m/s aur contact < 3 s.
- **Filter:** obstacle class + 55% + 0.5 s (danger 0.2 s). Baaki grey.
- **Speech:** urgent/normal slots, beep, interrupt, offline.
- **Camera lost → bolta hai.** Silence = khatarnak.
- **Currency:** transfer learning, MobileNetV3 / EfficientNet-B0 / ResNet50, no flip, label smoothing, AdamW + cosine. **Abhi trained nahi.**
- **Result:** 20 FPS, seated 3.4 → 0.84 m. Formal benchmark baaki hai.
