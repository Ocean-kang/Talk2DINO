import torch
import json
import argparse



def compute_i2t(
        image_features,
        text_features,
        image_ids,
        caption_to_image
):


    print("Computing Image-to-Text...")


    similarity = (
        image_features
        @
        text_features.T
    )


    recalls={}


    for k in [1,5,10]:

        hit=0


        topk=torch.topk(
            similarity,
            k=k,
            dim=1
        ).indices



        for i in range(
            len(image_ids)
        ):

            gt=image_ids[i]


            retrieved=topk[i]


            flag=False


            for idx in retrieved:


                if caption_to_image[
                    idx.item()
                ]==gt:

                    flag=True
                    break


            if flag:
                hit+=1



        recalls[k]=hit/len(image_ids)



    return recalls





def compute_t2i(

        image_features,
        text_features,
        image_ids,
        caption_to_image

):


    print(
        "Computing Text-to-Image..."
    )


    similarity=(

        text_features

        @

        image_features.T

    )



    recalls={}


    id_to_index={}

    for i,x in enumerate(image_ids):

        id_to_index[x]=i



    gt=[]

    for x in caption_to_image:

        gt.append(
            id_to_index[x]
        )



    gt=torch.tensor(gt)



    for k in [1,5,10]:


        hit=0


        topk=torch.topk(

            similarity,

            k=k,

            dim=1

        ).indices



        for i in range(
            len(gt)
        ):


            if gt[i] in topk[i]:

                hit+=1



        recalls[k]=hit/len(gt)



    return recalls





parser=argparse.ArgumentParser()

parser.add_argument(
    "--method",
    default="clip",
    choices=[
        "clip",
        "talk2dino"
    ]
)


args=parser.parse_args()



# =======================
# Load CLIP text feature
# =======================


clip=torch.load(

"outputs/retrieval/outputs_clip_retrieval.pt",

map_location="cpu"

)


clip["image_features"]=(
    clip["image_features"]
    .float()
)


clip["text_features"]=(
    clip["text_features"]
    .float()
)



if args.method=="clip":


    image_features=clip["image_features"]



else:


    talk=torch.load(

    "outputs/retrieval/outputs_talk2dino_retrieval.pt",

    map_location="cpu"

    )


    talk["image_features"]=(
        talk["image_features"]
        .float()
    )


    image_features=talk["image_features"]



text_features=clip["text_features"]


image_ids=clip["image_ids"]


caption_to_image=clip["caption_to_image"]



print(
"image:",
image_features.shape
)


print(
"text:",
text_features.shape
)



i2t=compute_i2t(

    image_features,

    text_features,

    image_ids,

    caption_to_image

)



t2i=compute_t2i(

    image_features,

    text_features,

    image_ids,

    caption_to_image

)



print("\n====================")

print(args.method)


print("\nImage-to-Text")

for k,v in i2t.items():

    print(
        f"R@{k}: {v*100:.3f}"
    )



print("\nText-to-Image")

for k,v in t2i.items():

    print(
        f"R@{k}: {v*100:.3f}"
    )


print("====================")
