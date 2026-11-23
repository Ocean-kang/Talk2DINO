# ECSD Next-Stage Results Analysis

## A. Candidate quality

### A1. Candidate-pool statistics

- Total candidate count:
- Mean candidates/image:
- Original / noun phrase / clause proportions:
- Heuristic invalid ratio:
- Duplicate ratio:
- Lexical unsupported-content proxy:
- Mean content-token count:
- Mean sentence-token count:

### A2. Selected-five statistics

- Selected candidate type proportions:
- Selected-caption heuristic invalid ratio:
- Selected-caption duplicate ratio:
- Selected-caption unsupported-content proxy:
- Mean content-token count:
- Mean sentence-token count:

### A3. Semantic-role coverage

| Role | Raw Caption count | Legacy candidate pool | Legacy selected five | ECSD v2 selected five |
|---|---:|---:|---:|---:|
| Entity | | | | |
| Attribute | | | | |
| Action | | | | |
| Relation | | | | |
| Context | | | | |
| Unknown | | | | |

### A4. Manual audit

- Sample seed / number of images:
- Candidate validity:
- Selected-caption validity:
- Unsupported-content rate:
- Semantic completeness:
- Role classification accuracy:
- Redundant vs complementary:
- Main failure modes:

## B. Selection ablation

| Arm | CISD | Semantic coverage | Category coverage | Redundancy | Heuristic validity |
|---|---:|---:|---:|---:|---:|
| B1 Coverage | | | | | |
| B2 Diversity | | | | | |
| B3 Coverage + Diversity | | | | | |
| B4 + Role Novelty | | | | | |

Questions:
1. Does B2 increase CISD without improving coverage?
2. Does B1 improve coverage at the cost of semantic dispersion?
3. Does B3 balance the two?
4. Does B4 add category coverage beyond B3?
5. Do human annotations agree with automatic candidate-quality proxies?

## C. Multi-seed OVSS

| Caption set | Seed | VOC20 | Context59 | COCO-Stuff | ADE20K |
|---|---:|---:|---:|---:|---:|
| Raw | 123 | | | | |
| Raw | 456 | | | | |
| Raw | 789 | | | | |
| Rich | 123 | | | | |
| Rich | 456 | | | | |
| Rich | 789 | | | | |
| Rich-Diversity | 123 | | | | |
| Rich-Diversity | 456 | | | | |
| Rich-Diversity | 789 | | | | |
| ECSD v2 | 123 | | | | |
| ECSD v2 | 456 | | | | |
| ECSD v2 | 789 | | | | |

Report mean ± population standard deviation, best, worst, relative improvement over Raw and Rich, and difference from Rich-Diversity.

## D. Interpretation

- Is the candidate pool linguistically usable?
- Does fact coverage contribute independently from CLIP diversity?
- Does role novelty add complementary information?
- Does caption-level CISD correlate with OVSS mIoU?
- Are effects consistent across seeds and datasets?
- Which failure modes remain?

Do not claim statistical significance from three seeds. Do not interpret lexical role labels as ground truth.
