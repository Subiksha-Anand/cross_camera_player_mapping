import json
import pandas as pd
import os

def iou(boxA, boxB):
    # Intersection over Union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def match_players(broadcast_json, tacticam_json, output_csv):
    with open(broadcast_json, 'r') as f:
        broadcast_data = json.load(f)

    with open(tacticam_json, 'r') as f:
        tacticam_data = json.load(f)

    # Convert to DataFrames
    df_b = pd.DataFrame(broadcast_data)
    df_t = pd.DataFrame(tacticam_data)

    # Filter player class (YOLO class 2 = player)
    df_b = df_b[df_b['confidence'] > 0.5]
    df_t = df_t[df_t['confidence'] > 0.5]

    df_b = df_b.reset_index(drop=True)
    df_t = df_t.reset_index(drop=True)

    rows = []
    player_id = 1

    for frame in sorted(set(df_b['frame']).intersection(df_t['frame'])):
        b_frame = df_b[df_b['frame'] == frame]
        t_frame = df_t[df_t['frame'] == frame]

        for _, b_row in b_frame.iterrows():
            b_box = b_row['bbox']
            best_iou = 0
            best_match = None

            for _, t_row in t_frame.iterrows():
                t_box = t_row['bbox']
                current_iou = iou(b_box, t_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_match = t_box

            if best_iou > 0.3:
                rows.append({
                    "frame": frame,
                    "player_id": player_id,
                    "tacticam_bbox": best_match,
                    "broadcast_bbox": b_box
                })
                player_id += 1

    # Save matches
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"[✅] Matched players saved to {output_csv}. Total: {len(df_out)} rows")

if __name__ == "__main__":
    match_players(
        broadcast_json="outputs/detections_broadcast.json",
        tacticam_json="outputs/detections_tacticam.json",
        output_csv="outputs/player_mappings.csv"
    )
