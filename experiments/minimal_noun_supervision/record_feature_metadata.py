"""Record an explicit attestation for legacy features after checking extraction logs.
This cannot infer a model checkpoint or resize setting from tensor dimensions.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.cache import model_metadata
from common.io import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-features', required=True)
    p.add_argument('--val-features', required=True)
    p.add_argument('--clip-checkpoint', required=True)
    p.add_argument('--dino-checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--confirm-source-models', action='store_true',
        help='Attest from your existing extraction records: B/16 CLIP, B/14 reg DINO, resize/crop 448')
    a = p.parse_args()
    if not a.confirm_source_models:
        p.error('Check original feature extraction settings/logs first; then pass --confirm-source-models')
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    m = model_metadata(a.clip_checkpoint, a.dino_checkpoint)
    m.update(train_features_sha256=sha256(a.train_features), val_features_sha256=sha256(a.val_features),
        source_model_attestation=True, verification='user-attested extraction provenance; not inferred from tensors')
    write_json(a.output, m)
    print(a.output)


if __name__ == '__main__':
    main()
