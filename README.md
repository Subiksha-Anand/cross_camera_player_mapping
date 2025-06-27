# Cross-Camera Football Player Mapping & Annotation

This project detects football players in two videos taken from different camera views (broadcast and tacticam), matches player identities across those views, and generates annotated videos with consistent player IDs.

---

## 📁 Project Structure

```

cross\_camera\_player\_mapping/
│
├── videos/                    # Input videos
│   ├── broadcast.mp4
│   └── tacticam.mp4
│
├── outputs/                   # Output files
│   ├── detections\_broadcast.json
│   ├── detections\_tacticam.json
│   ├── player\_mappings.csv
│   ├── broadcast\_annotated.mp4
│   └── tacticam\_annotated.mp4
│
├── model/
│   └── best.pt                # YOLOv8 model trained on football data
│
├── src/
│   ├── detect\_players.py      # Run YOLO detection on videos
│   ├── match\_players.py       # Match player IDs between views
│   ├── annotate\_broadcast.py  # Annotate broadcast video
│   ├── annotate\_tacticam.py   # Annotate tacticam video
│   └── utils.py               # Helper functions
│
├── README.md
├── REPORT.md
└── requirements.txt

````

---

## Environment Setup

### Dependencies

Install these Python packages before running the project:

```bash
pip install -r requirements.txt
````

###  `requirements.txt` content:

```txt
ultralytics>=8.0.0
opencv-python
pandas
numpy
scikit-learn
scipy
torch
torchvision
gdown
torchreid
```

---

##  How to Run the Project

> Make sure your `videos/` and `model/best.pt` exist before running the scripts.

### 1. **Detect players in both views**

```bash
python src/detect_players.py
```

This generates:

* `outputs/detections_broadcast.json`
* `outputs/detections_tacticam.json`

---

### 2. **Match players across both views**

```bash
python src/match_players.py
```

This generates:

* `outputs/player_mappings.csv`

---

### 3. **Annotate videos**

```bash
# Annotate broadcast video
python src/annotate_broadcast.py

# Annotate tacticam video
python src/annotate_tacticam.py
```

This saves annotated videos as:

* `outputs/broadcast_annotated.mp4`
* `outputs/tacticam_annotated.mp4`

---

## Output Preview

* Bounding boxes with labels like `Player #1`, `Player #2`, etc.
* Synchronized player IDs across both camera views
* Optional slow-down for better visualization

---

##  Model Info

* **YOLOv8 Model (`best.pt`)** trained on football player detection

* **Classes**:

  * `0`: Ball
  * `1`: Goalkeeper
  * `2`: Player
  * `3`: Referee

* **Re-identification**: Uses `torchreid` for improved player matching across different views

---

## Developed By

**Subiksha Anand**

