
import os
import argparse
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from utils.coco_utils import load_coco_images,get_image_path

device="cuda"

def main(args):
    os.makedirs(args.output,exist_ok=True)

    model=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14_reg")
    model.to(device).eval()

    images=load_coco_images(args.caption)

    transform=T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406),
                    (0.229,0.224,0.225))
    ])

    results={}

    with torch.no_grad():
        for image_id,name in tqdm(images.items()):
            path=get_image_path(args.image_dir,name)
            try:
                img=Image.open(path).convert("RGB")
            except:
                continue

            img=transform(img).unsqueeze(0).to(device)

            feat=model.forward_features(img)
            cls=feat["x_norm_clstoken"]

            results[image_id]=cls.squeeze(0).cpu()

    torch.save(results,os.path.join(args.output,"dino_cls.pt"))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--caption",
        default="data/coco2014/annotations/captions_train2014.json")
    parser.add_argument("--image_dir",
        default="data/coco2014/train2014")
    parser.add_argument("--output",
        default="outputs/local_cocu/dino_cls")
    args=parser.parse_args()
    main(args)
