
import torch


def recall_i2t(sim, image_ids, caption_to_image, k):

    top=torch.topk(
        sim,
        k=k,
        dim=1
    ).indices

    hit=0

    for i,row in enumerate(top):

        retrieved=[
            caption_to_image[j]
            for j in row
        ]

        if image_ids[i] in retrieved:
            hit+=1

    return hit/len(image_ids)



def recall_t2i(sim, caption_to_image, image_ids, k):

    top=torch.topk(
        sim,
        k=k,
        dim=1
    ).indices

    hit=0

    for i,row in enumerate(top):

        retrieved=[
            image_ids[j]
            for j in row
        ]

        if caption_to_image[i] in retrieved:
            hit+=1

    return hit/len(caption_to_image)



data_talk2dino=torch.load(
"./outputs/retrieval/outputs_talk2dino_retrieval.pt"
)
# data_talk2dino_cls=torch.load(
# "./outputs/retrieval/outputs_talk2dino_cls_retrieval.pt"
# )
# data_clip=torch.load(
# "./outputs/retrieval/outputs_clip_retrieval.pt"
# )
# data_clip["image_features"] = data_clip["image_features"].float()
# data_clip["text_features"] = data_clip["text_features"].float()

data = data_talk2dino
sim=data["image_features"] @ data["text_features"].T


for k in [1,5,10]:

    print(
        "Image-to-Text R@{}: {:.4f}".format(
            k,
            recall_i2t(
                sim,
                data["image_ids"],
                data["caption_to_image"],
                k
            )
        )
    )


sim2=sim.T


for k in [1,5,10]:

    print(
        "Text-to-Image R@{}: {:.4f}".format(
            k,
            recall_t2i(
                sim2,
                data["caption_to_image"],
                data["image_ids"],
                k
            )
        )
    )
