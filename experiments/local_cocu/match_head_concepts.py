import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import clip

from tqdm import tqdm


# ---------------------------------------------------------
# Make Talk2DINO repository root importable
#
# Talk2DINO/
# ├── src/
# └── experiments/
#     └── local_cocu/
#         └── match_head_concepts.py
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.model import ProjectionLayer


device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# IO
# ---------------------------------------------------------

def load_json(path):

    with open(
        path,
        "r"
    ) as f:

        return json.load(f)


# ---------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------

def normalize(x):

    return F.normalize(
        x,
        p=2,
        dim=-1
    )


# ---------------------------------------------------------
# Load Talk2DINO projector
# ---------------------------------------------------------

def load_projector(
        config_path,
        weight_path
):

    print(
        "Loading projector config:",
        config_path
    )

    projector = ProjectionLayer.from_config(
        config_path
    )

    print(
        "Loading projector weight:",
        weight_path
    )

    ckpt = torch.load(
        weight_path,
        map_location="cpu"
    )

    #
    # Support several checkpoint formats
    #

    if isinstance(ckpt, dict):

        if "state_dict" in ckpt:

            state_dict = ckpt[
                "state_dict"
            ]

        elif "model" in ckpt:

            state_dict = ckpt[
                "model"
            ]

        else:

            state_dict = ckpt

    else:

        state_dict = ckpt


    #
    # Remove possible DDP prefixes.
    #

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]

        if new_key.startswith(
            "projector."
        ):

            new_key = new_key[
                len("projector.") :
            ]

        cleaned[
            new_key
        ] = value


    #
    # IMPORTANT:
    #
    # Talk2DINO overrides load_state_dict()
    # but does NOT return _IncompatibleKeys.
    #
    # Therefore:
    #
    # WRONG:
    # missing, unexpected =
    #     projector.load_state_dict(...)
    #
    # RIGHT:
    #

    projector.load_state_dict(
        cleaned,
        strict=True
    )

    projector = projector.to(
        device
    )

    projector.eval()


    print(
        "Projector loaded successfully."
    )

    return projector


# ---------------------------------------------------------
# CLIP text encoder
# ---------------------------------------------------------

def encode_concepts(
        clip_model,
        projector,
        concepts,
        prompt_template
):

    #
    # Example:
    #
    # dog
    #
    # ->
    #
    # "a photo of dog"
    #

    texts = [
        prompt_template.format(
            concept
        )
        for concept in concepts
    ]


    tokens = clip.tokenize(
        texts
    ).to(
        device
    )


    with torch.no_grad():

        #
        # CLIP:
        #
        # [num_concepts, 512]
        #

        clip_text = clip_model.encode_text(
            tokens
        ).float()


        #
        # IMPORTANT:
        #
        # Talk2DINO maps:
        #
        # CLIP TEXT
        #     ↓
        # DINO FEATURE SPACE
        #
        # [C,512] -> [C,768]
        #

        projected_text = projector.project_clip_txt(
            clip_text
        )


    projected_text = normalize(
        projected_text.float()
    )


    return projected_text


# ---------------------------------------------------------
# Match one image
# ---------------------------------------------------------

def match_one_image(
        head_features,
        concept_features,
        concepts,
        topk,
        threshold
):

    #
    # head_features:
    #
    # [H, 768]
    #
    # concept_features:
    #
    # [C, 768]
    #

    head_features = normalize(
        head_features.float()
    )


    #
    # similarity:
    #
    # [H,C]
    #

    similarity = torch.matmul(
        head_features,
        concept_features.T
    )


    #
    # Each concept chooses the head
    # that matches it best.
    #
    # Concept c:
    #
    # max_h sim(head_h, concept_c)
    #

    scores, best_heads = similarity.max(
        dim=0
    )


    #
    # Optional threshold
    #

    if threshold is None:

        valid_indices = torch.arange(
            len(concepts),
            device=scores.device
        )

    else:

        valid_indices = torch.where(
            scores >= threshold
        )[0]


    if valid_indices.numel() == 0:

        return [], []


    valid_scores = scores[
        valid_indices
    ]


    k = min(
        topk,
        valid_indices.numel()
    )


    ranking = torch.topk(
        valid_scores,
        k=k
    ).indices


    selected_indices = valid_indices[
        ranking
    ]


    selected = []

    detailed = []


    for idx in selected_indices.tolist():

        concept = concepts[
            idx
        ]

        score = float(
            scores[idx].item()
        )

        head_id = int(
            best_heads[idx].item()
        )


        selected.append(
            concept
        )


        detailed.append(
            {
                "concept": concept,
                "score": score,
                "best_head": head_id
            }
        )


    return selected, detailed


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(args):

    os.makedirs(
        args.output,
        exist_ok=True
    )


    # -----------------------------------------------------
    # Candidate concepts
    # -----------------------------------------------------

    print(
        "Loading candidate concepts..."
    )

    candidates = load_json(
        args.candidate
    )


    # -----------------------------------------------------
    # Head features
    # -----------------------------------------------------

    print(
        "Loading head features..."
    )

    heads = torch.load(
        args.head_features,
        map_location="cpu"
    )


    print()
    print(
        "Candidate images:",
        len(candidates)
    )

    print(
        "Head-feature images:",
        len(heads)
    )


    # -----------------------------------------------------
    # Key sanity check
    # -----------------------------------------------------

    candidate_key = next(
        iter(candidates.keys())
    )

    head_key = next(
        iter(heads.keys())
    )


    print(
        "Example candidate key:",
        candidate_key,
        type(candidate_key)
    )

    print(
        "Example head key:",
        head_key,
        type(head_key)
    )


    nonempty_candidates = sum(
        bool(x)
        for x in candidates.values()
    )


    print(
        "Non-empty candidate images:",
        nonempty_candidates,
        "/",
        len(candidates)
    )


    # -----------------------------------------------------
    # CLIP
    # -----------------------------------------------------

    print()
    print(
        "Loading CLIP..."
    )

    clip_model, _ = clip.load(
        args.clip_model,
        device=device,
        jit=False
    )

    clip_model.eval()


    # -----------------------------------------------------
    # Talk2DINO
    # -----------------------------------------------------

    print(
        "Loading Talk2DINO projector..."
    )

    projector = load_projector(
        args.projector_config,
        args.weight
    )


    # -----------------------------------------------------
    # Matching
    # -----------------------------------------------------

    results = {}

    detailed_results = {}


    empty_candidates = 0
    missing_heads = 0
    processed = 0


    print()
    print(
        "Starting local concept matching..."
    )


    for image_id, concepts in tqdm(
        candidates.items()
    ):


        #
        # JSON:
        #
        # "57870"
        #
        # torch dict:
        #
        # 57870
        #

        try:

            image_id_int = int(
                image_id
            )

        except ValueError:

            results[
                image_id
            ] = []

            detailed_results[
                image_id
            ] = []

            continue


        # -------------------------------------------------
        # No concepts
        # -------------------------------------------------

        if not concepts:

            empty_candidates += 1

            results[
                image_id
            ] = []

            detailed_results[
                image_id
            ] = []

            continue


        # -------------------------------------------------
        # No visual feature
        # -------------------------------------------------

        if image_id_int not in heads:

            missing_heads += 1

            results[
                image_id
            ] = []

            detailed_results[
                image_id
            ] = []

            continue


        # -------------------------------------------------
        # Remove duplicate concepts
        # while preserving order
        # -------------------------------------------------

        concepts = list(
            dict.fromkeys(
                concepts
            )
        )


        # -------------------------------------------------
        # Encode candidate concepts
        #
        # CLIP:
        # 512
        #
        # Talk2DINO:
        # 512 -> 768
        # -------------------------------------------------

        concept_features = encode_concepts(
            clip_model,
            projector,
            concepts,
            args.prompt_template
        )


        # -------------------------------------------------
        # Visual local/head features
        # -------------------------------------------------

        visual_features = heads[
            image_id_int
        ]


        if not torch.is_tensor(
            visual_features
        ):

            visual_features = torch.tensor(
                visual_features
            )


        visual_features = visual_features.to(
            device
        ).float()


        #
        # Allow:
        #
        # [768]
        #
        # ->
        #
        # [1,768]
        #

        if visual_features.ndim == 1:

            visual_features = (
                visual_features
                .unsqueeze(0)
            )


        if visual_features.ndim != 2:

            raise RuntimeError(
                f"Image {image_id}: "
                f"unexpected visual feature shape "
                f"{visual_features.shape}"
            )


        # -------------------------------------------------
        # Dimension check
        # -------------------------------------------------

        if (
            visual_features.shape[-1]
            !=
            concept_features.shape[-1]
        ):

            raise RuntimeError(
                "\nFeature dimension mismatch!\n"
                f"image_id={image_id}\n"
                f"head feature="
                f"{visual_features.shape}\n"
                f"projected concept="
                f"{concept_features.shape}\n"
            )


        # -------------------------------------------------
        # Head-concept matching
        # -------------------------------------------------

        selected, detailed = match_one_image(
            visual_features,
            concept_features,
            concepts,
            args.topk,
            args.threshold
        )


        results[
            image_id
        ] = selected


        detailed_results[
            image_id
        ] = detailed


        processed += 1


    # -----------------------------------------------------
    # Save simple output
    # -----------------------------------------------------

    result_path = os.path.join(
        args.output,
        "matched_concepts.json"
    )


    with open(
        result_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )


    # -----------------------------------------------------
    # Save detailed output
    # -----------------------------------------------------

    detailed_path = os.path.join(
        args.output,
        "matched_concepts_detailed.json"
    )


    with open(
        detailed_path,
        "w"
    ) as f:

        json.dump(
            detailed_results,
            f,
            indent=2
        )


    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    nonempty_matches = sum(
        bool(x)
        for x in results.values()
    )


    print()
    print(
        "======================================"
    )

    print(
        "Local-CoCu matching completed"
    )

    print(
        "======================================"
    )


    print(
        "Total images:",
        len(candidates)
    )

    print(
        "Empty candidate images:",
        empty_candidates
    )

    print(
        "Missing head features:",
        missing_heads
    )

    print(
        "Processed:",
        processed
    )

    print(
        "Non-empty matched results:",
        nonempty_matches
    )

    print()
    print(
        "matched_concepts:"
    )

    print(
        result_path
    )

    print()
    print(
        "detailed:"
    )

    print(
        detailed_path
    )


# ---------------------------------------------------------
# Args
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--candidate",
        default=(
            "outputs/local_cocu/"
            "candidate/"
            "candidate_concepts.json"
        )
    )


    parser.add_argument(
        "--head_features",
        default=(
            "outputs/local_cocu/"
            "head_features/"
            "head_features.pt"
        )
    )


    parser.add_argument(
        "--projector_config",
        default=(
            "configs/"
            "vitb_mlp_infonce.yaml"
        )
    )


    parser.add_argument(
        "--weight",
        default=(
            "weights/"
            "vitb_mlp_infonce.pth"
        )
    )


    parser.add_argument(
        "--clip_model",
        default="ViT-B/16"
    )


    parser.add_argument(
        "--topk",
        type=int,
        default=10
    )


    parser.add_argument(
        "--threshold",
        type=float,
        default=None
    )


    parser.add_argument(
        "--prompt_template",
        default="a photo of {}"
    )


    parser.add_argument(
        "--output",
        default=(
            "outputs/local_cocu/"
            "matched"
        )
    )


    args = parser.parse_args()

    main(args)
