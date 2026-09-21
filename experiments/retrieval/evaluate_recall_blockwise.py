import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def recall_i2t_blockwise(
    image_features,
    text_features,
    image_ids,
    caption_to_image,
    ks,
    device,
    query_batch_size,
):
    max_k = max(ks)
    hits = {k: 0 for k in ks}

    # Keep candidates on GPU, but never construct the full N_image x N_caption matrix.
    text_features = text_features.to(device)
    caption_to_image = caption_to_image.to(device)

    for start in tqdm(
        range(0, len(image_features), query_batch_size),
        desc="Image -> Text",
    ):
        end = min(start + query_batch_size, len(image_features))

        q = image_features[start:end].to(device)
        sim = q @ text_features.T
        top = sim.topk(max_k, dim=1).indices

        retrieved_image_ids = caption_to_image[top]
        gt = image_ids[start:end].to(device).unsqueeze(1)

        for k in ks:
            hits[k] += (
                retrieved_image_ids[:, :k] == gt
            ).any(dim=1).sum().item()

    return {
        k: hits[k] / len(image_features)
        for k in ks
    }


@torch.no_grad()
def recall_t2i_blockwise(
    image_features,
    text_features,
    image_ids,
    caption_to_image,
    ks,
    device,
    query_batch_size,
):
    max_k = max(ks)
    hits = {k: 0 for k in ks}

    image_features = image_features.to(device)
    image_ids = image_ids.to(device)

    for start in tqdm(
        range(0, len(text_features), query_batch_size),
        desc="Text -> Image",
    ):
        end = min(start + query_batch_size, len(text_features))

        q = text_features[start:end].to(device)
        sim = q @ image_features.T
        top = sim.topk(max_k, dim=1).indices

        retrieved_image_ids = image_ids[top]
        gt = caption_to_image[start:end].to(device).unsqueeze(1)

        for k in ks:
            hits[k] += (
                retrieved_image_ids[:, :k] == gt
            ).any(dim=1).sum().item()

    return {
        k: hits[k] / len(text_features)
        for k in ks
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="outputs/retrieval/outputs_talk2dino_cls_retrieval.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=256)
    args = parser.parse_args()

    data = torch.load(args.features, map_location="cpu")

    image_features = data["image_features"].float()
    text_features = data["text_features"].float()

    # Normalize defensively even if the extraction script already did it.
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    image_ids = torch.as_tensor(data["image_ids"], dtype=torch.long)
    caption_to_image = torch.as_tensor(
        data["caption_to_image"],
        dtype=torch.long,
    )

    if len(image_ids) != len(image_features):
        raise RuntimeError("image_ids length != number of image features")

    if len(caption_to_image) != len(text_features):
        raise RuntimeError(
            "caption_to_image length != number of text features"
        )

    if len(torch.unique(image_ids)) != len(image_ids):
        raise RuntimeError(
            "Duplicate image_ids detected. Retrieval image candidates "
            "must contain one feature per unique image."
        )

    valid_ids = set(image_ids.tolist())
    missing = [
        x for x in set(caption_to_image.tolist())
        if x not in valid_ids
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} caption image IDs are missing from image candidates."
        )

    print("features:", args.features)
    print("image_features:", tuple(image_features.shape))
    print("text_features:", tuple(text_features.shape))
    print("num unique images:", len(image_ids))
    print("num captions:", len(caption_to_image))

    if "metadata" in data:
        print("metadata:", data["metadata"])

    ks = (1, 5, 10)

    i2t = recall_i2t_blockwise(
        image_features,
        text_features,
        image_ids,
        caption_to_image,
        ks,
        args.device,
        args.query_batch_size,
    )

    t2i = recall_t2i_blockwise(
        image_features,
        text_features,
        image_ids,
        caption_to_image,
        ks,
        args.device,
        args.query_batch_size,
    )

    print("\n===== Retrieval Recall =====")
    for k in ks:
        print(f"Image-to-Text R@{k}: {i2t[k] * 100:.2f}")
    for k in ks:
        print(f"Text-to-Image R@{k}: {t2i[k] * 100:.2f}")

    mr = (
        sum(i2t.values()) + sum(t2i.values())
    ) / (len(i2t) + len(t2i))
    print(f"Mean Recall: {mr * 100:.2f}")


if __name__ == "__main__":
    main()
