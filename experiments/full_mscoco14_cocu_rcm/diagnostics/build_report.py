#!/usr/bin/env python3
"""Create a factual Full-vs-5K diagnostic report from saved experiment artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def load_csv(path):
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fmt(value):
    if value is None or value == "":
        return "待运行"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", default="outputs/full_mscoco14_cocu_rcm/work")
    p.add_argument("--output-dir", default="outputs/full_mscoco14_cocu_rcm")
    p.add_argument("--five-k-work", default="outputs/5k_cocu_rcm/work")
    p.add_argument("--five-k-summary", default="outputs/5k_cocu_rcm/summary.csv")
    a = p.parse_args()
    work, out = Path(a.work_dir), Path(a.output_dir)

    concept = load_json(work / "concept_archive" / "concept_statistics.json")
    concept_frequency = load_json(work / "concept_archive" / "concept_frequency.json")
    region = load_json(work / "region_verification" / "verification_statistics.json")
    sampling = load_json(work / "sampling_statistics.json")
    training = load_csv(out / "summary.csv")
    five_k = load_csv(a.five_k_summary)
    five_k_work = Path(a.five_k_work)
    five_k_concept = load_json(five_k_work / "concept_archive" / "concept_statistics.json")
    five_k_region = load_json(five_k_work / "region_verification" / "verification_statistics.json")
    five_k_sampling = load_json(five_k_work / "sampling_statistics.json")

    lines = [
        "# Full MSCOCO14 CoCu-RCM 规模诊断报告",
        "",
        "本报告只汇总磁盘上已经存在的统计与结果，不对未运行的阶段填入推测值。",
        "",
        "## 1. Concept Archive",
        "",
        f"- Images: {fmt(concept.get('num_images'))}",
        f"- Images with concepts: {fmt(concept.get('images_with_concepts'))}",
        f"- Images without concepts: {fmt(concept.get('images_without_concepts'))}",
        f"- Concept mentions: {fmt(concept.get('concept_mentions'))}",
        f"- Unique normalized concepts: {fmt(concept.get('unique_normalized_concepts'))}",
        "",
        "### Top-15 CoCu concepts by image frequency",
        "",
        "| Concept | Images |",
        "|---|---:|",
    ]
    for name, count in list(concept_frequency.items())[:15]:
        lines.append(f"| {name} | {count} |")
    if not concept_frequency:
        lines.append("| 待运行 | |")
    lines += [
        "",
        "## 2. 5K vs Full: Concept Statistics",
        "",
        "| Metric | 5K | Full |",
        "|---|---:|---:|",
        f"| Images | {fmt(five_k_concept.get('num_images'))} | {fmt(concept.get('num_images'))} |",
        f"| Images with concepts | {fmt(five_k_concept.get('images_with_concepts'))} | {fmt(concept.get('images_with_concepts'))} |",
        f"| Concept mentions | {fmt(five_k_concept.get('concept_mentions'))} | {fmt(concept.get('concept_mentions'))} |",
        f"| Unique concepts | {fmt(five_k_concept.get('unique_normalized_concepts'))} | {fmt(concept.get('unique_normalized_concepts'))} |",
        "",
        "## 3. 5K vs Full: Region Verification",
        "",
        "| Metric | 5K | Full |",
        "|---|---:|---:|",
        f"| Verified image-concept pairs | {fmt(five_k_region.get('image_concept_pairs_verified'))} | {fmt(region.get('image_concept_pairs_verified'))} |",
        f"| Verified ratio | {fmt(five_k_region.get('verified_ratio'))} | {fmt(region.get('verified_ratio'))} |",
        f"| Verified region mentions | {fmt(five_k_region.get('verified_region_mentions'))} | {fmt(region.get('verified_region_mentions'))} |",
        f"| Mean crop similarity | {fmt(five_k_region.get('crop_similarity', {}).get('mean'))} | {fmt(region.get('crop_similarity', {}).get('mean'))} |",
        f"| Mean local gain | {fmt(five_k_region.get('local_gain', {}).get('mean'))} | {fmt(region.get('local_gain', {}).get('mean'))} |",
        "",
        "## 4. 5K vs Full: K-shot Sampling",
        "",
        "| Metric | 5K | Full |",
        "|---|---:|---:|",
        f"| Total anchors | {fmt(five_k_sampling.get('total_anchors'))} | {fmt(sampling.get('total_anchors'))} |",
        f"| Same-image coverage | {fmt(five_k_sampling.get('same_coverage'))} | {fmt(sampling.get('same_coverage'))} |",
        f"| Cross-image coverage | {fmt(five_k_sampling.get('cross_coverage'))} | {fmt(sampling.get('cross_coverage'))} |",
        f"| Shuffled coverage | {fmt(five_k_sampling.get('shuffled_coverage'))} | {fmt(sampling.get('shuffled_coverage'))} |",
        "",
        "## 5. Full Region Verification Details",
        "",
        f"- Images processed: {fmt(region.get('images_processed'))}",
        f"- Verified image-concept pairs: {fmt(region.get('image_concept_pairs_verified'))}",
        f"- Verified region mentions: {fmt(region.get('verified_region_mentions'))}",
        f"- Unique verified concepts: {fmt(region.get('unique_verified_concepts'))}",
        f"- Mean crop similarity: {fmt(region.get('crop_similarity', {}).get('mean'))}",
        f"- Mean local gain: {fmt(region.get('local_gain', {}).get('mean'))}",
        f"- Mean top1-top2 margin: {fmt(region.get('top1_top2_margin', {}).get('mean'))}",
        "",
        "## 6. Full K-shot Sampling",
        "",
        f"- Total anchors: {fmt(sampling.get('total_anchors'))}",
        f"- Same-image eligible: {fmt(sampling.get('same_eligible'))}",
        f"- Same-image coverage: {fmt(sampling.get('same_coverage'))}",
        f"- Cross-image eligible: {fmt(sampling.get('cross_eligible'))}",
        f"- Cross-image coverage: {fmt(sampling.get('cross_coverage'))}",
        f"- Shuffled eligible: {fmt(sampling.get('shuffled_eligible'))}",
        f"- Shuffled coverage: {fmt(sampling.get('shuffled_coverage'))}",
        f"- RCM/Shuffled anchor sets identical: {fmt(sampling.get('rcm_shuffled_anchor_set_identical'))}",
        f"- RCM/Shuffled source IDs identical: {fmt(sampling.get('rcm_shuffled_source_ids_identical'))}",
        f"- RCM/Shuffled negative IDs identical: {fmt(sampling.get('rcm_shuffled_negative_ids_identical'))}",
        "",
        "### Per-frequency-bucket coverage",
        "",
        "| Bucket | Anchors | Same | Cross | Shuffled |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket_name, bucket in sampling.get("frequency_buckets", {}).items():
        lines.append(
            f"| {bucket_name} | {fmt(bucket.get('total'))} | "
            f"{fmt(bucket.get('same_coverage'))} | {fmt(bucket.get('cross_coverage'))} | "
            f"{fmt(bucket.get('shuffled_coverage'))} |"
        )
    if not sampling.get("frequency_buckets"):
        lines.append("| 待运行 | | | | |")
    lines += [
        "",
        "## 7. Full OVSS Results",
        "",
        "| Arm | Seed | VOC20 | PC59/Context59 | COCO-Stuff | ADE20K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in training:
        lines.append(
            f"| {row.get('arm','')} | {row.get('seed','')} | "
            f"{row.get('voc20_mIoU','')} | {row.get('context59_mIoU','')} | "
            f"{row.get('coco_stuff_mIoU','')} | {row.get('ade20k_mIoU','')} |"
        )
    if not training:
        lines.append("| 待运行 | | | | | |")
    by_seed_arm = {(str(row.get("seed")), str(row.get("arm"))): row for row in training}
    lines += [
        "",
        "### Paired CoCu-RCM minus CoCu-RCM-Shuffled",
        "",
        "| Seed | VOC20 Δ | PC59 Δ | COCO-Stuff Δ | ADE20K Δ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in sorted({str(row.get("seed")) for row in training}):
        rcm = by_seed_arm.get((seed, "cocu_rcm"), {})
        shuffled = by_seed_arm.get((seed, "cocu_rcm_shuffled"), {})
        deltas = []
        for key in ("voc20_mIoU", "context59_mIoU", "coco_stuff_mIoU", "ade20k_mIoU"):
            try:
                deltas.append(f"{float(rcm[key]) - float(shuffled[key]):+.4f}")
            except (KeyError, TypeError, ValueError):
                deltas.append("待运行")
        lines.append(f"| {seed} | " + " | ".join(deltas) + " |")
    if not training:
        lines.append("| 待运行 | | | | |")
    lines += [
        "",
        "## 8. Training and Gradient Diagnostics",
        "",
        "| Arm | Seed | Global Loss | Local Raw | Local Scaled | Weighted Local | Local/Global Grad |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in training:
        lines.append(
            f"| {row.get('arm','')} | {row.get('seed','')} | "
            f"{row.get('final_train_global_loss','')} | {row.get('final_train_local_raw','')} | "
            f"{row.get('final_train_local_scaled','')} | {row.get('final_weighted_local_loss','')} | "
            f"{row.get('final_local_global_grad_ratio','')} |"
        )
    if not training:
        lines.append("| 待运行 | | | | | | |")
    lines += [
        "",
        "## 9. 5K Reference",
        "",
        "如果提供了 `outputs/5k_cocu_rcm/summary.csv`，下表列出其已有记录；"
        "否则请先运行 5K 结果收集脚本或手动提供 summary.csv。",
        "",
        "| Arm | Seed | VOC20 | Context59 | COCO-Stuff | ADE20K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in five_k:
        lines.append(
            f"| {row.get('arm','')} | {row.get('seed','')} | "
            f"{row.get('voc20_mIoU','')} | {row.get('context59_mIoU','')} | "
            f"{row.get('coco_stuff_mIoU','')} | {row.get('ade20k_mIoU','')} |"
        )
    if not five_k:
        lines.append("| 未提供 | | | | | |")
    lines += [
        "",
        "## 10. Interpretation Checklist",
        "",
        "- 比较 Full 与 5K 的概念频率分布，确认高频概念是否主导跨图像正样本。",
        "- 比较 Same 与 RCM 的有效 Anchor 覆盖率，避免只看 mIoU 而忽略样本选择偏差。",
        "- RCM 与 Shuffled 必须使用相同 Anchor、来源图像 ID 和负样本 ID；若三个一致性字段不是 true，不能将性能差异归因于概念对应关系。",
        "- 检查 Local/Global Loss 和梯度范数，判断 Full 数据规模是否改变局部监督的相对强度。",
        "- 只有多 Seed 结果稳定后，才讨论 RCM 是否真正受益于更大数据规模。",
        "",
        "## 11. Data Availability",
        "",
        f"- Training summary rows: {len(training)}",
        f"- 5K reference rows: {len(five_k)}",
        "- 缺少的结果保持为待运行，不做外推。",
    ]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "docs" / "full_mscoco14_cocu_rcm_analysis.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[REPORT] {path}")


if __name__ == "__main__":
    main()
