import sys

sys.path.append(
    "/home/master/code/oymk/Talking2dino/Talk2DINO"
)


import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from tqdm import tqdm


from coco_dataset import COCORetrievalDataset, retrieval_collate_fn


from src.model import ProjectionLayer



device="cuda"



# ============================
# DINOv2
# ============================

print("Loading DINOv2")


dino=torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14"
)


dino.cuda()
dino.eval()



# ============================
# CLIP text projector
# ============================

print("Loading ProjectionLayer")


projector=ProjectionLayer(
    dino_embed_dim=768,
    clip_embed_dim=512,
    hidden_layer=1

)


ckpt=torch.load(
    "weights/vitb_mlp_infonce.pth",
    map_location="cpu"
)


print(
    "checkpoint:",
    ckpt.keys()
)


projector.load_state_dict(
    ckpt,
    strict=True
)


projector.cuda()
projector.eval()



# ============================
# Image transform
# ============================


transform=T.Compose(
[
    T.Resize(224),
    T.CenterCrop(224),

    T.ToTensor(),

    T.Normalize(
        (0.485,0.456,0.406),
        (0.229,0.224,0.225)
    )
]
)



dataset=COCORetrievalDataset(

    "data/coco2014/val2014",

    "data/coco2014/annotations/captions_val2014.json",

    transform

)



loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=8,
    collate_fn=retrieval_collate_fn
)



features=[]

image_ids=[]



with torch.no_grad():

    for batch in tqdm(loader):


        images=batch["image"].cuda()



        # DINO feature

        feat=dino(images)


        feat=feat.float()


        feat/=feat.norm(
            dim=-1,
            keepdim=True
        )



        features.append(
            feat.cpu()
        )


        image_ids.extend(
            batch["image_id"]
        )



features=torch.cat(
    features
)



print(
"image feature:",
features.shape
)



torch.save(

{

"image_features":
features,

"image_ids":
image_ids

},

"outputs/retrieval/outputs_talk2dino_retrieval.pt"

)


print("saved")