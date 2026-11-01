"""Gallery-defined lexical concepts; standard library only, no POS parser."""
from collections import Counter
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def repo_module(relative):
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError('Required existing repository helper: ' + str(path))
    spec = importlib.util.spec_from_file_location('mns_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_gallery(path):
    import json
    helper = repo_module('experiments/local_cocu/utils/concept_utils.py')
    lexical = repo_module('experiments/local_cocu_novel/filter_novel.py')
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    supported = helper.normalize_gallery(obj)  # reuse existing format validator
    arr = obj.get('concept', list(obj)) if isinstance(obj, dict) else obj
    if not isinstance(arr, list):
        raise ValueError('Gallery must be list, {concept:list}, or concept-keyed dictionary')
    names, seen, aliases = [], set(), []
    for item in arr:
        raw = item.get('concept') if isinstance(item, dict) else item
        if not isinstance(raw, str) or raw not in supported:
            raise ValueError('Unsupported or invalid gallery entry: ' + repr(item))
        name = ' '.join(lexical.tokenize(raw))
        if not name:
            raise ValueError('Empty normalized gallery entry: ' + repr(raw))
        if name in seen:
            aliases.append({'entry': raw, 'normalized': name, 'reason': 'duplicate_normalized_surface'})
            continue
        seen.add(name)
        names.append(name)
        if name != raw:
            aliases.append({'entry': raw, 'normalized': name, 'reason': 'case_punctuation_whitespace'})
    if len(names) < 2:
        raise ValueError('Need at least two unique gallery concepts')
    return names, aliases


class GalleryMatcher:
    """Bounded token-trie scan. Matches stay inside one original caption."""
    def __init__(self, gallery, morphology='exact'):
        self.gallery = gallery
        self.lexical = repo_module('experiments/local_cocu_novel/filter_novel.py')
        self.morphology = morphology
        forms = None
        if morphology == 'repo_forms':
            forms = repo_module('experiments/Head_caption_S5_Editing/editing.py').forms
        self.trie = {}
        for gid, concept in enumerate(gallery):
            surfaces = [(concept, False)] if forms is None else forms(concept)
            for surface, plural in surfaces:
                node = self.trie
                for word in self.lexical.tokenize(surface):
                    node = node.setdefault(word, {})
                node.setdefault(None, []).append((gid, surface, plural))

    def extract(self, captions):
        counts, evidence = Counter(), []
        for caption_index, caption in enumerate(captions):
            words, found = self.lexical.tokenize(caption), set()
            for start in range(len(words)):
                node = self.trie
                for end in range(start, len(words)):
                    node = node.get(words[end])
                    if node is None:
                        break
                    for gid, surface, plural in node.get(None, []):
                        # Existing exact phrase matcher, with broad stemming disabled.
                        assert self.lexical.in_raw_captions(surface, [caption], exact_only=True)
                        found.add(gid)
                        evidence.append({'gallery_id': gid, 'concept': self.gallery[gid],
                            'caption_index': caption_index, 'token_start': start, 'token_end': end + 1,
                            'matched_surface': ' '.join(words[start:end + 1]),
                            'rule': 'repository_plural_form' if plural else 'exact_normalized_phrase'})
            counts.update(found)
        ids = sorted(counts, key=lambda gid: (-counts[gid], self.gallery[gid]))
        return {'concepts': [self.gallery[g] for g in ids],
                'caption_frequency': {self.gallery[g]: counts[g] for g in ids},
                'gallery_ids': ids, 'matches': evidence, 'no_match': not bool(ids)}


def compose_prompt(concepts, fits):
    # Same deterministic greedy whole-concept rule as the previous version.
    kept, removed = [], []
    for concept in concepts:
        candidate = 'a photo of ' + ', '.join(kept + [concept])
        if fits(candidate):
            kept.append(concept)
        else:
            removed.append(concept)
    if not kept:
        raise ValueError('No complete concept fits the CLIP text context')
    return 'a photo of ' + ', '.join(kept), kept, removed
