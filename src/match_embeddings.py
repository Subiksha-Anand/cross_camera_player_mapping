import json
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import os

def match_embeddings(broadcast_path, tacticam_path, output_csv):
    with open(broadcast_path, 'r') as f:
        broadcast = json.load(f)

    with open(tacticam_path, 'r') as f:
        tacticam = json.load(f)

    df_b = pd.DataFrame(broadcast)
    df_t = pd.DataFrame(tacticam)

    matches = []
    player_id = 1

    for frame in sorted(set(df_b["frame"]).intersection(df_t["frame"])):
        b_frame = df_b[df_b["frame"] == frame]
        t_frame = df_t[df_t["frame"] == frame]

        for _, b_row in b_frame.iterrows():
            b_emb = np.array(b_row["embedding"])
            best_sim = 1.0
            best_row = None

            for _, t_row in t_frame.iterrows():
                t_emb = np.array(t_row["embedding"])
                sim = cosine(b_emb, t_emb)
                if sim < best_sim:
                    best_sim = sim
                    best_row = t_row

            if best_sim < 0.5:  # good match
                matches.append({
                    "frame": frame,
                    "player_id": player_id,
                    "tacticam_bbox": best_row["bbox"],
                    "broadcast_bbox": b_row["bbox"]
                })
                player_id += 1

    df_out = pd.DataFrame(matches)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"[✅] Matched players using embeddings saved to {output_csv}. Total: {len(df_out)}")

if __name__ == "__main__":
    match_embeddings(
        broadcast_path="outputs/embeddings_broadcast.json",
        tacticam_path="outputs/embeddings_tacticam.json",
        output_csv="outputs/player_mappings.csv"
    )
