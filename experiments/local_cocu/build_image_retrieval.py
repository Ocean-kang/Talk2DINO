
import torch
import argparse
import json
import os
from tqdm import tqdm

def normalize(x):
    return x/x.norm(dim=-1,keepdim=True)

def main(args):
    os.makedirs(args.output,exist_ok=True)

    data=torch.load(args.feature)
    ids=list(data.keys())

    feats=torch.stack([data[i] for i in ids])
    feats=normalize(feats)

    sim=feats@feats.T

    result={}

    for i,img_id in enumerate(tqdm(ids)):
        idx=torch.topk(sim[i],args.topk+1).indices
        result[str(img_id)]=[
            ids[j.item()] for j in idx
            if ids[j.item()]!=img_id
        ][:args.topk]

    with open(os.path.join(args.output,"neighbors.json"),"w") as f:
        json.dump(result,f,indent=2)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--feature",
        default="outputs/local_cocu/dino_cls/dino_cls.pt")
    parser.add_argument("--topk",type=int,default=50)
    parser.add_argument("--output",
        default="outputs/local_cocu/retrieval")
    args=parser.parse_args()
    main(args)
