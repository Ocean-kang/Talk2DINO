# Data Manifest and Integrity Contract

## Input files

| Input | Expected fields | Integrity checks |
|---|---|---|
| Full Train PTH | `images`, `annotations`, image `disentangled_self_attn`, annotation `ann_feats` | 82,783 unique IDs; exactly 5 captions/image; 12x768 head features; 512-D CLIP features |
| Full Val PTH | `images`, `annotations`, image `avg_self_attn_out`, annotation `ann_feats` | 40,504 unique IDs; exactly 5 captions/image; train/val disjoint |
| Local-CoCu matches | image ID -> concept list | exact Train2014 ID coverage; explicit empty list allowed; no missing IDs |
| CoCu discovery head features | `raw_train.pth` image `disentangled_self_attn` `[12,768]` by default; optional external `image_id -> [12,768]` cache | exact Train2014 ID coverage; finite tensor shape |
| COCO image files | `file_name` from annotation/PTH | resolved before extraction |
| DINO/CLIP weights | existing local caches/checkpoints | missing caches cause a hard error; no backbone download is initiated |

## Artifact hashes

Each stage records source/config hashes:
- `data_manifest.json`: source Raw PTH SHA256.
- `concept_archive/manifest.json`: Raw PTH, CoCu matches, Train ID-set and shard SHA256.
- `discovery/manifest.json`: captions JSON, head-feature source and hash, gallery, Teacher weights, config, settings and discovery shard SHA256.
- `region_gallery_manifest.json`: Raw PTH, Concept manifest, settings, feature index and feature/metadata shard SHA256.
- `verification_manifest.json`: Concept manifest, Region manifest, CLIP weights, settings and verified-shard SHA256.
- Sampling `manifest.json`: augmented PTH, verification manifest, region index/manifest, settings and JSONL shard SHA256.
- `train_*.resume.pt`: train/val/config/sampling/region manifest hashes.

## Split leakage prevention

- Only Train2014 IDs are included in Concept Archive, Region Gallery, verification and sampling.
- Val2014 IDs are checked disjoint from Train2014 and used only by the Talk2DINO validation Dataset.
- The sampling builder reads only the Train augmented PTH and Train-derived verified Region Gallery.
- RCM positive source image IDs are selected from the Train concept inverted index.
- No Val PTH is passed to concept discovery, region extraction, verification or sampling.

## Explicit empty vs missing

- A full concept archive must contain one record for every Train2014 image.
- An image with no selected CoCu concepts is stored with `status="explicit_empty"` and `concepts=[]`.
- A missing image record is a hard error.
- Images with no concepts are skipped by default Region Scope and listed in the Region Gallery manifest.
