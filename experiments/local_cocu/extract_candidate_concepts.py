import os
import json
import argparse

from tqdm import tqdm

from utils.coco_utils import load_coco_captions
from utils.concept_utils import (
    extract_words,
    match_gallery
)


def load_gallery(path):

    with open(path,"r") as f:
        return json.load(f)



def main(args):

    os.makedirs(
        args.output,
        exist_ok=True
    )


    #
    # neighbors
    #
    with open(
        args.neighbors,
        "r"
    ) as f:
        neighbors=json.load(f)



    #
    # captions
    #
    captions=load_coco_captions(
        args.caption
    )


    #
    # gallery
    #
    with open(
        args.gallery,
        "r"
    ) as f:

        gallery=json.load(f)



    results={}



    for image_id, retrieved_ids in tqdm(
        neighbors.items()
    ):

        concepts=[]


        #
        # 遍历topK retrieved images
        #
        for rid in retrieved_ids:


            if int(rid) not in captions:
                continue


            for cap in captions[
                int(rid)
            ]:


                words=extract_words(
                    cap
                )


                words=match_gallery(
                    words,
                    gallery
                )


                concepts.extend(
                    words
                )



        #
        # 去重
        #
        concepts=list(
            set(concepts)
        )


        results[
            image_id
        ]=concepts



    with open(
        os.path.join(
            args.output,
            "candidate_concepts.json"
        ),
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )



if __name__=="__main__":


    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--neighbors",
        default=
        "outputs/local_cocu/retrieval/neighbors.json"
    )


    parser.add_argument(
        "--caption",
        default=
        "data/coco2014/annotations/captions_train2014.json"
    )


    parser.add_argument(
        "--gallery",
        default=
        "experiments/local_cocu/MSCOCO14_train_gallery.json"
    )


    parser.add_argument(
        "--output",
        default=
        "outputs/local_cocu/candidate"
    )


    args=parser.parse_args()


    main(args)