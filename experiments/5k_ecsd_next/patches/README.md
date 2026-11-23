# Multi-seed patch

The original repository calls `set_seed(123)` inside `train_and_eval()` and does not expose a seed CLI argument. `src/train_util.py::do_train()` already accepts a seed, so this patch forwards an optional `--seed` value and seeds DataLoader workers.

From the Talk2DINO repository root:

```bash
bash experiments/5k_ecsd_next/patches/apply_seed_patch.sh
python train.py --help | grep seed
```

The default remains `123`, preserving the original command-line default. The model, projector, loss, optimizer recipe, number of epochs, and inference code are unchanged. The patch only changes seed plumbing and explicit DataLoader worker seeding.

If your local files differ from the checked GitHub revision, inspect `train_seed.patch` and apply the equivalent minimal edits manually. Do not force-apply a patch to a modified training file.
