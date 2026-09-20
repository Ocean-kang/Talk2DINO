"""CoCu caption augmentation dataset for Talk2DINO."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import Dataset



class CocuDinoClipDataset(Dataset):

    """
    COCO captions + additional CoCu captions.

    COCO captions are always kept.

    cocu_ratio controls additional CoCu captions.

    Example:

    original image:

        5 COCO captions


    cocu_ratio=0:

        5 COCO


    cocu_ratio=1:

        5 COCO
        +
        5 CoCu


    cocu_ratio=2:

        5 COCO
        +
        10 CoCu

    """


    def __init__(
            self,
            base_features_file,
            curated_file=None,
            cocu_ratio=0.0,
            features_name="disentangled_self_attn",
            text_features="ann_feats",
            load_attn_maps=False,
    ):


        if cocu_ratio < 0:
            raise ValueError(
                "cocu_ratio must >=0"
            )


        data=torch.load(
            base_features_file,
            map_location="cpu"
        )


        self.images={
            int(im["id"]):im
            for im in data["images"]
        }


        self.image_ids=list(
            self.images.keys()
        )


        self.original=defaultdict(list)


        for ann in data["annotations"]:

            image_id=int(
                ann["image_id"]
            )

            if text_features not in ann:
                raise KeyError(
                    f"Missing {text_features}"
                )


            self.original[image_id].append(
                ann
            )



        self.curated=defaultdict(list)


        if curated_file:

            sidecar=torch.load(
                curated_file,
                map_location="cpu"
            )


            for ann in sidecar["annotations"]:

                if "ann_feats" not in ann:
                    raise KeyError(
                        "Curated annotation missing ann_feats"
                    )


                self.curated[
                    int(ann["image_id"])
                ].append(
                    ann
                )


        if cocu_ratio>0 and curated_file is None:

            raise ValueError(
                "curated_file required"
            )



        # ==========================
        # 关键改变
        # 预先展开caption sample
        # ==========================


        self.samples=[]


        for image_id in self.image_ids:


            coco_caps=self.original[image_id]


            # ---------------------
            # 1. COCO 永远加入
            # ---------------------

            for ann in coco_caps:

                self.samples.append(
                    {
                        "image_id":image_id,
                        "ann":ann,
                        "source":"coco"
                    }
                )



            # ---------------------
            # 2. 增加CoCu
            # ---------------------

            if image_id not in self.curated:
                continue


            curated_caps=self.curated[image_id]


            num_add=int(
                len(coco_caps)
                *
                cocu_ratio
            )


            selected=curated_caps[:num_add]


            for ann in selected:

                self.samples.append(
                    {
                        "image_id":image_id,
                        "ann":ann,
                        "source":"cocu"
                    }
                )



        self.features_name=features_name

        self.text_features=text_features

        self.load_attn_maps=load_attn_maps



    def __len__(self):

        return len(self.samples)



    def __getitem__(self,index):


        item=self.samples[index]


        image_id=item["image_id"]

        ann=item["ann"]


        image=self.images[image_id]



        result={

            "annotation":
                ann["ann_feats"],


            "image":
                image[self.features_name],


            "metadata":
            {
                "annotation_id":
                    int(ann["id"]),

                "image_id":
                    image_id,

                "source":
                    item["source"],
            }

        }



        if self.load_attn_maps:

            result["self_attn_maps"] = (
                image["self_attn_maps"]
            )

            result["dino_features"] = (
                image["dino_features"]
            )


        return result