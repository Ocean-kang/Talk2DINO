
import sys
sys.path.append("/home/master/code/oymk/Talking2dino/Talk2DINO")

import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.model import ProjectionLayer
from coco_dataset import COCORetrievalDataset, retrieval_collate_fn


device="cuda"


dino=torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14"
).to(device)

dino.eval()


clip_model, _ = clip.load(
    "ViT-B/16",
    device=device
)

clip_model.eval()


projector=ProjectionLayer.from_config(
    "configs/vitb_mlp_infonce.yaml"
)


ckpt=torch.load(
    "weights/vitb_mlp_infonce.pth",
    map_location="cpu"
)

if "state_dict" in ckpt:
    ckpt=ckpt["state_dict"]
elif "model" in ckpt:
    ckpt=ckpt["model"]

projector.load_state_dict(
    ckpt,
    strict=False
)

projector.to(device)
projector.eval()


transform=T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        (0.485,0.456,0.406),
        (0.229,0.224,0.225)
    )
])


dataset=COCORetrievalDataset(
    "data/coco2014/val2014",
    "data/coco2014/annotations/captions_val2014.json",
    transform
)


loader=DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    collate_fn=retrieval_collate_fn
)


image_features=[]
text_features=[]
image_ids=[]
caption_to_image=[]


with torch.no_grad():

    for batch in tqdm(loader):

        images=batch["image"].to(device)

        img=dino.forward_features(images)
        img=img["x_norm_clstoken"]

        img=torch.nn.functional.normalize(
            img,
            dim=-1
        )

        image_features.append(
            img.cpu()
        )

        image_ids.extend(
            batch["image_id"]
        )


        texts=[]
        ids=[]

        for i,caps in enumerate(batch["captions"]):
            texts.extend(caps)
            ids.extend(
                [batch["image_id"][i]]*len(caps)
            )


        token=clip.tokenize(texts).to(device)

        txt=clip_model.encode_text(token)
        txt=projector.project_clip_txt(txt)
        txt=torch.nn.functional.normalize(
            txt,
            dim=-1
        )


        text_features.append(
            txt.cpu()
        )

        caption_to_image.extend(ids)



torch.save(
{
"image_features":torch.cat(image_features),
"text_features":torch.cat(text_features),
"image_ids":image_ids,
"caption_to_image":caption_to_image
},
"outputs_talk2dino_retrieval.pt"
)

print("saved")
