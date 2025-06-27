# annotate_tacticam.py
import cv2
import pandas as pd
import ast
import os

def annotate_video(video_path, mapping_csv, save_path):
    print("[📥] Loading player mappings...")
    df = pd.read_csv(mapping_csv)

    # Convert bbox strings to lists safely
    def safe_eval(val):
        try:
            return ast.literal_eval(val)
        except:
            print(f"[⚠️] Skipping row due to parse error → {val}")
            return None

    df["tacticam_bbox"] = df["tacticam_bbox"].apply(safe_eval)
    df = df.dropna(subset=["tacticam_bbox"])

    if "frame" not in df.columns or "player_id" not in df.columns:
        print("[❌] CSV format is incorrect. 'frame' or 'player_id' column missing.")
        return

    print("[🎥] Opening tacticam video...")
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[📏] Video resolution: {width}x{height}, FPS: {fps:.2f}")
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_matches = df[df["frame"] == frame_id]
        for _, row in frame_matches.iterrows():
            bbox = row["tacticam_bbox"]
            pid = row["player_id"]
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {pid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    print(f"[✅] Annotated video saved to {save_path}")

if __name__ == "__main__":
    annotate_video(
        video_path="videos/tacticam.mp4",
        mapping_csv="outputs/player_mappings.csv",
        save_path="outputs/annotated_tacticam.mp4"
    )
