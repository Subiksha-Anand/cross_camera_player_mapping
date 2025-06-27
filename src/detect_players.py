from ultralytics import YOLO
import cv2
import os
import json

def detect_and_save(video_path, model_path, save_dir, view_name):
    os.makedirs(save_dir, exist_ok=True)

    model = YOLO(model_path)
    print(model.names)  # See the class names

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id == 0:
            cv2.imwrite(f"frame_preview_{view_name}.jpg", frame)
            print(f"[🖼️] Frame 0 saved as frame_preview_{view_name}.jpg")

        # Detect only class 2 (players)
        results = model(frame, conf=0.25, verbose=False)[0]
        if not results.boxes:
            print(f"[⚠️] Frame {frame_id}: No detections")
        else:
            print(f"[✅] Frame {frame_id}: {len(results.boxes)} boxes found")

        for box in results.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            if cls != 2:  # Only keep 'player' detections
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "frame": frame_id,
                "view": view_name,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })
            print(f"   ↳ Player detected at {x1,y1,x2,y2} (conf: {conf:.2f})")

        frame_id += 1
        if frame_id > 50:  # Optional: limit for testing
            break

    cap.release()

    # Save to JSON
    json_path = os.path.join(save_dir, f"detections_{view_name}.json")
    with open(json_path, 'w') as f:
        json.dump(detections, f)
    print(f"[💾] Detections saved to {json_path}")

# Run both videos
if __name__ == "__main__":
    detect_and_save(
        video_path=r"C:\Users\subik\OneDrive\Document\PROJECT\cross_camera_player_mapping\videos\broadcast.mp4",
        model_path="../model/best.pt",
        save_dir="../outputs",
        view_name="broadcast"
    )
    detect_and_save(
        video_path=r"C:\Users\subik\OneDrive\Document\PROJECT\cross_camera_player_mapping\videos\tacticam.mp4",
        model_path="../model/best.pt",
        save_dir="../outputs",
        view_name="tacticam"
    )
