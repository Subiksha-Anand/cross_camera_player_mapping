import cv2
import pandas as pd
import ast
import os

def annotate_video(
    video_path,
    csv_path,
    output_path,
    speed_factor=0.5,  # set < 1.0 to slow down
    show_preview=False
):
    # Load mapping data
    df = pd.read_csv(csv_path)

    # Parse bboxes safely
    def safe_eval(val):
        try:
            return ast.literal_eval(val)
        except Exception as e:
            print(f"[⚠️] Skipping row due to parse error: {val} → {e}")
            return None

    df["broadcast_bbox"] = df["broadcast_bbox"].apply(safe_eval)
    df = df[df["broadcast_bbox"].notna()]  # remove bad rows

    # Convert 'frame' column to int if needed
    df["frame"] = df["frame"].astype(int)

    print(f"[📄] Loaded {len(df)} valid player mappings")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[❌] Failed to open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) * speed_factor
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_id = 0
    print(f"[🎥] Starting annotation...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_matches = df[df["frame"] == frame_id]

        if not frame_matches.empty:
            print(f"[📸] Annotating frame {frame_id} with {len(frame_matches)} player(s)")
            for _, row in frame_matches.iterrows():
                x1, y1, x2, y2 = row["broadcast_bbox"]
                player_id = row["player_id"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Player {player_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if show_preview:
            cv2.imshow("Annotated", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()
    print(f"[✅] Annotated video saved to {output_path}")


if __name__ == "__main__":
    annotate_video(
        video_path="videos/broadcast.mp4",
        csv_path="outputs/player_mappings.csv",
        output_path="outputs/annotated_video.mp4",
        speed_factor=0.5,           # 0.5x speed (slower)
        show_preview=False          # set True if you want to watch live
    )
