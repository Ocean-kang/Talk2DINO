import hashlib
import math
import torch
from torch.utils.data import Dataset, Sampler


def prompt_index(seed, epoch, image_id, count):
    key = ('%d:%d:%d' % (seed, epoch, image_id)).encode('ascii')
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % count


class PromptPoolDataset(Dataset):
    """One image per pass; prompt choice independent of DataLoader worker count."""
    def __init__(self, records, images, bank, seed=123, feature_name='disentangled_self_attn', fixed=False):
        self.records = records
        self.images = images
        self.bank = bank['embeddings']
        self.lookup = {t: i for i, t in enumerate(bank['texts'])}
        self.seed, self.feature_name, self.fixed, self.epoch = seed, feature_name, fixed, 0
        ids = [int(r['image_id']) for r in records]
        if not records or len(ids) != len(set(ids)):
            raise ValueError('Prompt pools must be nonempty and have unique image IDs')
        for rec in records:
            iid = int(rec['image_id'])
            if not rec['prompt_pool'] or iid not in images or feature_name not in images[iid]:
                raise ValueError('Missing prompt/image/feature for image %s' % iid)
            for text in rec['prompt_pool']:
                if text not in self.lookup:
                    raise ValueError('Missing text embedding: ' + text)
        for iid in ids:
            feat = images[iid][feature_name]
            if not torch.isfinite(feat).all():
                raise ValueError('Non-finite image feature: %s' % iid)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        iid = int(rec['image_id'])
        p = 0 if self.fixed else prompt_index(self.seed, self.epoch, iid, len(rec['prompt_pool']))
        text = rec['prompt_pool'][p]
        return {'annotation': self.bank[self.lookup[text]], 'image': self.images[iid][self.feature_name],
                'metadata': {'image_id': iid, 'annotation_id': idx}}


class ImageBatches(Sampler):
    """Unique image indices per batch; optionally truncate the final image pass."""
    def __init__(self, size, batch_size, seed, epoch, max_batches=None):
        if min(size, batch_size) <= 0:
            raise ValueError('Empty data or invalid batch size')
        self.size, self.batch_size, self.seed, self.epoch = size, batch_size, seed, epoch
        full = math.ceil(size / batch_size)
        self.count = min(full, max_batches) if max_batches is not None else full
        if self.count < 1:
            raise ValueError('Need at least one batch')

    def __len__(self):
        return self.count

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for i in range(self.count):
            yield order[i * self.batch_size:(i + 1) * self.batch_size]


def reference_budget(annotation_count, batch_size, epochs):
    if min(annotation_count, batch_size, epochs) <= 0:
        raise ValueError('Training budget must be positive')
    return math.ceil(annotation_count / batch_size) * epochs
