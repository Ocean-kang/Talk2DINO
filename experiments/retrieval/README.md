
# Talk2DINO COCO Retrieval

放置:
Talk2DINO/experiments/retrieval/

运行:
1. 修改 extract_talk2dino.py 中 projector checkpoint key(如果需要)
2. 提取CLIP:
CUDA_VISIBLE_DEVICES=0 python experiments/retrieval/extract_clip.py

3. 提取Talk2DINO:
CUDA_VISIBLE_DEVICES=0 python experiments/retrieval/extract_talk2dino.py \
 --weight weights/vitb_mlp_infonce_cocu_p50.pth

4. 评测:
python experiments/retrieval/evaluate_recall.py
