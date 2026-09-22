
import json
import os

def load_coco_images(caption_json):
    with open(caption_json, "r") as f:
        data = json.load(f)
    return {x["id"]: x["file_name"] for x in data["images"]}

def load_coco_captions(caption_json):
    with open(caption_json, "r") as f:
        data = json.load(f)
    captions = {}
    for ann in data["annotations"]:
        captions.setdefault(ann["image_id"], []).append(ann["caption"])
    return captions

def get_image_path(root, filename):
    return os.path.join(root, filename)
