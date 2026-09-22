# TODO: generate local_cocu_compare.json
import os
import json
import argparse

from tqdm import tqdm


# ---------------------------------------------------------
# JSON utils
# ---------------------------------------------------------

def load_json(path):

    with open(
        path,
        "r"
    ) as f:

        return json.load(f)



def save_json(
        data,
        path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )



# ---------------------------------------------------------
# COCO caption loader
# ---------------------------------------------------------

def load_coco_captions(
        path
):

    data = load_json(
        path
    )


    image_to_caption = {}


    #
    # COCO format:
    #
    # {
    #   images:[
    #       {"id":xxx}
    #   ],
    #
    #   annotations:[
    #       {
    #        image_id:xxx,
    #        caption:"..."
    #       }
    #   ]
    # }
    #

    for ann in data["annotations"]:

        image_id = str(
            ann["image_id"]
        )

        caption = ann[
            "caption"
        ]


        if image_id not in image_to_caption:

            image_to_caption[
                image_id
            ] = []


        image_to_caption[
            image_id
        ].append(
            caption
        )


    return image_to_caption



# ---------------------------------------------------------
# Build curated caption
# ---------------------------------------------------------

def build_caption(
        concepts
):

    if len(concepts)==0:

        return ""


    return (
        "a photo containing "
        +
        " ".join(
            concepts
        )
    )



# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(args):


    print(
        "Loading captions..."
    )

    captions = load_coco_captions(
        args.caption_json
    )


    print(
        "Loading retrieval..."
    )

    neighbors = load_json(
        args.neighbors
    )


    print(
        "Loading candidate concepts..."
    )

    candidates = load_json(
        args.candidates
    )


    print(
        "Loading matched concepts..."
    )

    matched = load_json(
        args.matched
    )


    print(
        "Loading detailed matching..."
    )

    detailed = load_json(
        args.detailed
    )


    results = {}


    print(
        "Building Local-CoCu compare..."
    )


    for image_id in tqdm(
        candidates.keys()
    ):


        item = {}


        # ---------------------------------------------
        # image id
        # ---------------------------------------------

        item[
            "image_id"
        ] = int(
            image_id
        )


        # ---------------------------------------------
        # original caption
        # ---------------------------------------------

        item[
            "original_caption"
        ] = captions.get(
            image_id,
            []
        )


        # ---------------------------------------------
        # retrieved images
        # ---------------------------------------------

        retrieved = neighbors.get(
            image_id,
            []
        )


        item[
            "retrieved_images"
        ] = [
            int(x)
            for x in retrieved
        ]



        # ---------------------------------------------
        # candidate concepts
        # ---------------------------------------------

        item[
            "candidate_concepts"
        ] = candidates.get(
            image_id,
            []
        )



        # ---------------------------------------------
        # curated concepts
        # ---------------------------------------------

        item[
            "curated_concepts"
        ] = matched.get(
            image_id,
            []
        )


        # ---------------------------------------------
        # curated caption
        # ---------------------------------------------

        item[
            "curated_caption"
        ] = build_caption(
            item[
                "curated_concepts"
            ]
        )



        # ---------------------------------------------
        # head level analysis
        # ---------------------------------------------

        item[
            "concept_head_scores"
        ] = detailed.get(
            image_id,
            []
        )



        results[
            image_id
        ] = item



    save_json(
        results,
        args.output
    )


    print()
    print(
        "================================"
    )

    print(
        "Local-CoCu compare generated"
    )

    print(
        "================================"
    )

    print(
        "images:",
        len(results)
    )

    print(
        "output:"
    )

    print(
        args.output
    )



# ---------------------------------------------------------
# Args
# ---------------------------------------------------------

if __name__=="__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--caption_json",
        default=(
            "data/coco2014/"
            "annotations/"
            "captions_train2014.json"
        )
    )


    parser.add_argument(
        "--neighbors",
        default=(
            "outputs/local_cocu/"
            "retrieval/"
            "neighbors.json"
        )
    )


    parser.add_argument(
        "--candidates",
        default=(
            "outputs/local_cocu/"
            "candidate/"
            "candidate_concepts.json"
        )
    )


    parser.add_argument(
        "--matched",
        default=(
            "outputs/local_cocu/"
            "matched/"
            "matched_concepts.json"
        )
    )


    parser.add_argument(
        "--detailed",
        default=(
            "outputs/local_cocu/"
            "matched/"
            "matched_concepts_detailed.json"
        )
    )


    parser.add_argument(
        "--output",
        default=(
            "outputs/local_cocu/"
            "results/"
            "local_cocu_compare.json"
        )
    )


    args = parser.parse_args()


    main(args)