# 📝 Project Report: Cross-Camera Football Player Mapping & Annotation

## Objective

The goal of this project was to detect and consistently annotate football players across two video feeds taken from different camera angles—**broadcast** and **tacticam**—using object detection and identity matching techniques.

---

##  Approach & Methodology

### 1. **Object Detection with YOLOv8**

- Utilized a custom-trained `YOLOv8` model (`best.pt`) capable of identifying:
  - Players
  - Goalkeepers
  - Referees
  - Football (ball)

- Detected bounding boxes for each player on a per-frame basis using:
  ```python
  from ultralytics import YOLO
  model = YOLO('model/best.pt')
````

* Videos processed:

  * `broadcast.mp4`
  * `tacticam.mp4`

* Saved outputs as:

  * `detections_broadcast.json`
  * `detections_tacticam.json`

---

### 2. **Cross-View Player Matching**

* Used frame-wise bounding box comparison between both camera views.
* Early technique: **IOU + coordinate matching** for simple overlap estimation.
* Upgraded to **torchreid (Re-Identification)** for robust player identity preservation using appearance features.
* Output: `player_mappings.csv` containing consistent player IDs across views.

---

### 3. **Video Annotation**

* Used OpenCV to draw bounding boxes and label each player with a unique ID.
* Two scripts:

  * `annotate_broadcast.py`
  * `annotate_tacticam.py`
* Annotations dynamically linked to frames and bounding box data from `player_mappings.csv`.
* Videos saved as:

  * `broadcast_annotated.mp4`
  * `tacticam_annotated.mp4`

---

## 🧪 Experiments & Techniques Tried

| Technique               | Description                           | Outcome                                 |
| ----------------------- | ------------------------------------- | --------------------------------------- |
| YOLOv8n pretrained      | Basic detection model                 | Missed players due to low resolution    |
| Custom YOLOv8 model     | Trained on football data              | Good performance for player detection   |
| IOU Matching            | Basic matching by overlap             | Failed under perspective shift          |
| TorchReID Matching      | Re-ID model using appearance features | High consistency across views           |
| Cropping tacticam frame | Cropped to center to better focus     | Slight improvement in detection         |
| Slow motion videos      | Slowed down playback using OpenCV     | Helped in visual clarity of annotations |

---

##  Challenges Encountered

1. **Tacticam video quality**

   * Players appear smaller and distant.
   * Required adjusting crop and resolution for improved detection.

2. **Unmatched frame timing**

   * Frames in `broadcast` and `tacticam` are not always aligned.
   * Led to difficulty in matching corresponding detections.

3. **Low-confidence detections**

   * Some players were detected with very low confidence scores.
   * Resolved by lowering YOLO confidence threshold (`conf=0.1`).

4. **Incomplete annotation**

   * Some annotations were skipped due to frame mismatch or failed parsing in `player_mappings.csv`.
   * Fixed with robust error handling and filtering of malformed data.

---

##  Incomplete Work

* **Frame Synchronization**:
  Full synchronization between views (e.g., aligning kickoff, goals) is not implemented.
  ➤ Would require optical flow or timestamp-based methods.

* **Team Classification**:
  Players are not classified into teams based on jersey color.
  ➤ Could use a K-Means clustering of jersey colors from cropped images.

* **Real-time Application**:
  The pipeline is currently offline. A real-time pipeline using threads and video capture APIs is feasible but not implemented.

---

## If Given More Time / Resources

* Integrate jersey color detection for team-based grouping.
* Use Deep SORT or ByteTrack for continuous player tracking.
* Train YOLO with more diverse tactical and aerial footage to improve generalization.
* Use frame interpolation to align both videos more accurately for matching.
* Package the system as a Streamlit app for easy demo and interaction.

---

## Final Output

* `detections_broadcast.json` and `detections_tacticam.json`: raw detections
* `player_mappings.csv`: matched player identities
* `broadcast_annotated.mp4` and `tacticam_annotated.mp4`: annotated output videos

---

## Prepared by:

**Subiksha Anand**

```