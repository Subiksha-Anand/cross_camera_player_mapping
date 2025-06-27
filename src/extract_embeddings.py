import os
import json
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model = resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove final classifier
model.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_embeddings(video_path, detections_path, save_path, view_name):
    with open(detections_path, 'r') as f:
        detections = json.load(f)

    cap = cv2.VideoCapture(video_path)
    frame_dict = {}

    # Pre-index detections by frame
    for det in detections:
        frame_dict.setdefault(det["frame"], []).append(det)

    results = []

    for frame_id in tqdm(sorted(frame_dict.keys())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        for det in frame_dict[frame_id]:
            x1, y1, x2, y2 = map(int, det["bbox"])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            try:
                img_tensor = transform(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(img_tensor).squeeze().cpu().numpy()
                    emb = emb / np.linalg.norm(emb)
            except Exception as e:
                continue

            results.append({
                "frame": det["frame"],
                "bbox": det["bbox"],
                "embedding": emb.tolist(),
                "view": view_name
            })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"[✅] Saved embeddings for {view_name} to {save_path}")

# Run for both views
if __name__ == "__main__":
    extract_embeddings(
        video_path="videos/broadcast.mp4",
        detections_path="outputs/detections_broadcast.json",
        save_path="outputs/embeddings_broadcast.json",
        view_name="broadcast"
    )
    extract_embeddings(
        video_path="videos/tacticam.mp4",
        detections_path="outputs/detections_tacticam.json",
        save_path="outputs/embeddings_tacticam.json",
        view_name="tacticam"
    )
