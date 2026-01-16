import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import clip

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp", "jfif"}

WASTE_SCENE_LABELS = [
    "a photo of a garbage dump",
    "a photo of a landfill",
    "a photo of waste pile",
    "a polluted area with trash",
    "overflowing garbage"
]

CLEAN_SCENE_LABELS = [
    "a clean park",
    "a clean street",
    "a clean outdoor area"
]

HIGH_WASTE_THRESHOLD = 0.30  

yolo = YOLO("yolov8n.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)



def load_image_array(path):
    return Image.open(path).convert("RGB")

def is_image(path):
    return path.split(".")[-1].lower() in IMAGE_EXTENSIONS


def clip_scene_analysis(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(WASTE_SCENE_LABELS + CLEAN_SCENE_LABELS).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)

        similarity = (image_features @ text_features.T).softmax(dim=-1)

    waste_score = similarity[0][:len(WASTE_SCENE_LABELS)].sum().item()
    return waste_score
def yolo_object_analysis(image):
    img_array = np.array(image)
    results = yolo(img_array, verbose=False)
    return len(results[0].boxes)

def detect_waste(image_path):
    image = load_image_array(image_path)

    yolo_count = yolo_object_analysis(image)
    clip_score = clip_scene_analysis(image)

    if clip_score > HIGH_WASTE_THRESHOLD:
        decision = "HIGH WASTE AREA"
    elif yolo_count > 3:
        decision = "MODERATE WASTE AREA"
    else:
        decision = "AREA APPEARS CLEAN"

    return {
        "decision": decision,
        "yolo_objects_detected": yolo_count,
        "clip_waste_score": round(clip_score, 2)
    }


if __name__ == "__main__":
    path = input("Enter image path from local storage: ").strip()

    if not os.path.exists(path) or not is_image(path):
        print("Invalid image file.")
        exit()

    result = detect_waste(path)

    print("\nAI Waste Analysis Result")
    print("------------------------")
    print("Decision:", result["decision"])
    print("YOLO Objects Detected:", result["yolo_objects_detected"])
    print("CLIP Waste Score:", result["clip_waste_score"])
