import sys
sys.path.append("/home/master/code/oymk/Talking2dino/Talk2DINO")


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import CLIP.clip as clip


from coco_dataset import COCORetrievalDataset



device="cuda"



model, preprocess = clip.load(
    "ViT-B/16",
    device=device
)

model.eval()



dataset = COCORetrievalDataset(
    "data/coco2014/val2014",
    "data/coco2014/annotations/captions_val2014.json",
    preprocess
)



loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    collate_fn=lambda x:x
)



image_features=[]
text_features=[]

image_ids=[]
caption_to_image=[]



with torch.no_grad():

    for batch in tqdm(loader):

        # =====================
        # image
        # =====================

        images=torch.stack(
            [
                x["image"]
                for x in batch
            ]
        ).to(device)



        img_feat=model.encode_image(
            images
        )


        img_feat /= img_feat.norm(
            dim=-1,
            keepdim=True
        )


        image_features.append(
            img_feat.cpu()
        )


        image_ids.extend(
            [
                x["image_id"]
                for x in batch
            ]
        )


        # =====================
        # text
        # =====================

        texts=[]
        ids=[]


        for sample in batch:

            caps=sample["captions"]

            texts.extend(
                caps
            )


            ids.extend(
                [sample["image_id"]]
                *
                len(caps)
            )



        token=clip.tokenize(
            texts
        ).to(device)



        txt_feat=model.encode_text(
            token
        )


        txt_feat /= txt_feat.norm(
            dim=-1,
            keepdim=True
        )


        text_features.append(
            txt_feat.cpu()
        )


        caption_to_image.extend(
            ids
        )



torch.save(
{
    "image_features":
        torch.cat(image_features),

    "text_features":
        torch.cat(text_features),

    "image_ids":
        image_ids,

    "caption_to_image":
        caption_to_image
},

"outputs_clip_retrieval.pt"

)