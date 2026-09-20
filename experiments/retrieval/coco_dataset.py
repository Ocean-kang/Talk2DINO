import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class COCORetrievalDataset(Dataset):

    def __init__(
        self,
        image_dir,
        caption_json,
        transform=None
    ):

        self.image_dir = image_dir
        self.transform = transform


        with open(caption_json, "r") as f:
            data = json.load(f)


        self.images = {}

        for x in data["images"]:
            self.images[x["id"]] = x["file_name"]


        self.captions = {}

        for ann in data["annotations"]:

            self.captions.setdefault(
                ann["image_id"],
                []
            ).append(
                ann["caption"]
            )


        self.ids = list(
            self.captions.keys()
        )


    def __len__(self):

        return len(self.ids)


    def __getitem__(self, idx):

        image_id = self.ids[idx]


        path = os.path.join(
            self.image_dir,
            self.images[image_id]
        )


        img = Image.open(path).convert("RGB")


        if self.transform:
            img = self.transform(img)


        return {
            "image": img,
            "image_id": image_id,
            "captions": self.captions[image_id]
        }



def retrieval_collate_fn(batch):

    images = []
    image_ids = []
    captions = []


    for item in batch:

        images.append(
            item["image"]
        )

        image_ids.append(
            item["image_id"]
        )

        captions.append(
            item["captions"]
        )


    return {

        "image":
            torch.stack(images, dim=0),

        "image_id":
            image_ids,

        "captions":
            captions

    }