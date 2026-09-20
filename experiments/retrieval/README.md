
# Talk2DINO COCO14 Retrieval Final Version

适配:
- CLIP baseline
- Talk2DINO ProjectionLayer (CLIP text -> DINO space)
- vitb_mlp_infonce.pth

目录:
放到:
Talk2DINO/experiments/retrieval/

执行:

1. 提取CLIP:
CUDA_VISIBLE_DEVICES=0 python experiments/retrieval/extract_clip.py

2. 提取Talk2DINO:
CUDA_VISIBLE_DEVICES=0 python experiments/retrieval/extract_talk2dino.py \
--weight weights/vitb_mlp_infonce.pth

3. 计算Recall:
python experiments/retrieval/evaluate_recall.py
