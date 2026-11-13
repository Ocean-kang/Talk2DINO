"""Pure-Python RA-C2C algorithm. No LLM, spaCy, or heavyweight NLP dependency."""
from __future__ import annotations
import re
from collections import defaultdict
from itertools import chain

TOKEN_RE = re.compile(r"[a-z0-9]+")
IRREGULAR = {
    "men": "man", "women": "woman", "people": "person", "children": "child",
    "mice": "mouse", "geese": "goose", "feet": "foot", "teeth": "tooth",
    "bikes": "bike", "bicycles": "bicycle", "cars": "car", "dogs": "dog",
    "cats": "cat", "birds": "bird", "horses": "horse", "boys": "boy",
    "girls": "girl", "trees": "tree", "flowers": "flower", "buildings": "building",
}
STOP_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "with", "and", "or",
    "is", "are", "was", "were", "be", "being", "been", "this", "that",
    "there", "here", "for", "from", "by", "as", "into", "over", "under",
    "photo", "picture", "image", "showing", "shows", "some", "many",
}
SCENE_TERMS = {
    "street", "beach", "kitchen", "park", "forest", "stadium", "room", "field",
    "mountain", "ocean", "sea", "city", "indoor", "outdoor", "restaurant",
    "airport", "bathroom", "bedroom", "living room", "landscape", "sky",
    "river", "lake", "pool", "garden", "farm", "classroom", "office",
    "snow", "road", "highway", "coast", "desert", "bridge",
}
ATTRIBUTE_TERMS = {
    "red", "blue", "green", "yellow", "black", "white", "brown", "orange",
    "pink", "purple", "gray", "grey", "golden", "silver", "dark", "light",
    "small", "large", "big", "little", "tall", "short", "long", "young",
    "old", "wooden", "metal", "metallic", "plastic", "glass", "leather",
    "striped", "spotted", "smooth", "rough", "shiny", "dirty", "clean",
    "colorful", "bright", "round", "square", "rectangular", "furry",
}
ACTION_FORMS = {
    "riding": "riding", "rides": "riding", "rode": "riding",
    "holding": "holding", "holds": "holding", "held": "holding",
    "eating": "eating", "eats": "eating", "ate": "eating",
    "playing": "playing", "plays": "playing", "played": "playing",
    "walking": "walking", "walks": "walking", "walked": "walking",
    "running": "running", "runs": "running", "ran": "running",
    "standing": "standing", "stands": "standing", "stood": "standing",
    "sitting": "sitting", "sits": "sitting", "sat": "sitting",
    "jumping": "jumping", "jumps": "jumping", "jumped": "jumping",
    "carrying": "carrying", "carries": "carrying", "carried": "carrying",
    "driving": "driving", "drives": "driving", "drove": "driving",
    "looking": "looking", "looks": "looking", "looked": "looking",
    "throwing": "throwing", "throws": "throwing", "threw": "throwing",
    "catching": "catching", "catches": "catching", "caught": "catching",
    "cutting": "cutting", "cuts": "cutting", "cooking": "cooking",
    "cooked": "cooking", "sleeps": "sleeping", "sleeping": "sleeping",
    "swimming": "swimming", "swims": "swimming", "surfing": "surfing",
    "flying": "flying", "flies": "flying", "wearing": "wearing",
    "wears": "wearing", "wear": "wearing", "baking": "baking",
    "bouncing": "bouncing", "leaning": "leaning", "kicking": "kicking",
    "painting": "painting", "talking": "talking", "smiling": "smiling",
}
RELATIONS = (
    ("in front of", "in front of"), ("next to", "next to"),
    ("across from", "across from"), ("left of", "to the left of"),
    ("right of", "to the right of"), ("above", "above"), ("below", "below"),
    ("behind", "behind"), ("beside", "beside"), ("under", "under"),
    ("inside", "inside"), ("near", "near"), ("on", "on"), ("over", "over"),
    ("at", "at"),
)
SOURCE_WEIGHTS = {"f5": 1.0, "cocu": 0.9, "local_cocu": 0.9, "bootstrap": 0.8, "p1": 1.0}

def tokens(text: str) -> list[str]:
    out = []
    for word in TOKEN_RE.findall(str(text).lower()):
        word = IRREGULAR.get(word, word)
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes")):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "ous")):
            word = word[:-1]
        out.append(word)
    return out

def norm_phrase(text: str) -> tuple[str, ...]:
    return tuple(tokens(text))

def contains_phrase(text: str, phrase: str) -> bool:
    hay, needle = tokens(text), tokens(phrase)
    if not needle or len(needle) > len(hay):
        return False
    return any(hay[i:i + len(needle)] == needle for i in range(len(hay) - len(needle) + 1))

def phrase_mentions(text: str, phrases: list[str]) -> list[dict]:
    """Longest-first non-overlapping phrase matching with lightweight morphology."""
    words = tokens(text)
    lex, max_len = {}, 1
    for phrase in phrases:
        key = norm_phrase(phrase)
        if key:
            lex.setdefault(key, phrase)
            max_len = max(max_len, len(key))
    found, occupied = [], set()
    for start in range(len(words)):
        if start in occupied:
            continue
        hit = None
        for size in range(min(max_len, len(words) - start), 0, -1):
            key = tuple(words[start:start + size])
            if key in lex:
                hit = (size, lex[key])
                break
        if hit:
            size, phrase = hit
            found.append({"concept": phrase, "start": start, "end": start + size})
            occupied.update(range(start, start + size))
    return found

def rank_normalize(rows: list[dict]) -> dict[str, float]:
    """Convert one source's ordering/scores to [0,1]; never mix raw score scales."""
    if not rows:
        return {}
    def value(row):
        for key in ("score", "target_score", "confidence", "residual_score", "cosine"):
            try:
                x = float(row.get(key))
                if x == x and abs(x) != float("inf"):
                    return x
            except (TypeError, ValueError):
                pass
        return None
    vals = [(value(r), i, str(r.get("concept", "")).strip()) for i, r in enumerate(rows)]
    if any(v is not None for v, _, _ in vals):
        ordered = sorted(vals, key=lambda x: (-(x[0] if x[0] is not None else float("-inf")), x[1], x[2]))
    else:
        ordered = sorted(vals, key=lambda x: (x[1], x[2]))
    out, denom = {}, max(1, len(ordered) - 1)
    for rank, (_, _, concept) in enumerate(ordered):
        if concept:
            out.setdefault(concept, 1.0 - rank / denom)
    return out

def classify_concept(concept: str) -> str:
    phrase = norm_phrase(concept)
    if not phrase:
        return "unknown"
    joined = " ".join(phrase)
    if joined in {" ".join(norm_phrase(x)) for x in SCENE_TERMS}:
        return "scene"
    if len(phrase) == 1 and phrase[0] in ATTRIBUTE_TERMS:
        return "attribute"
    if len(phrase) == 1 and phrase[0] in ACTION_FORMS:
        return "action"
    return "object"

def article(phrase: str) -> str:
    p = phrase.strip()
    if not p:
        return p
    if p.lower().startswith(("a ", "an ", "the ", "some ")):
        return p
    if p.lower() in {"people", "children", "men", "women"}:
        return p
    return ("an " if p.split()[0].lower()[:1] in "aeiou" else "a ") + p

def lexical_overlap(a: str, b: str) -> float:
    aa = {x for x in tokens(a) if x not in STOP_WORDS}
    bb = {x for x in tokens(b) if x not in STOP_WORDS}
    return len(aa & bb) / max(1, len(aa | bb)) if aa and bb else 0.0

def _find_actions(captions: list[str], object_terms: list[str]) -> list[dict]:
    records = []
    for cap_idx, caption in enumerate(captions):
        words = tokens(caption)
        mentions = phrase_mentions(caption, object_terms)
        for idx, word in enumerate(words):
            if word not in ACTION_FORMS:
                continue
            before = [m for m in mentions if m["end"] <= idx and idx - m["end"] <= 6]
            after = [m for m in mentions if m["start"] > idx and m["start"] - idx <= 7]
            subject = before[-1]["concept"] if before else None
            target = after[0]["concept"] if after else None
            if subject:
                records.append({"action": ACTION_FORMS[word], "subject": subject,
                                "target": target, "caption_index": cap_idx})
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["subject"], row["action"], row["target"])].append(row["caption_index"])
    out = []
    for (subject, action, target), support_ids in grouped.items():
        out.append({"subject": subject, "action": action, "target": target,
                    "support": len(set(support_ids)), "support_ids": sorted(set(support_ids))})
    return sorted(out, key=lambda x: (-x["support"], x["subject"], x["action"], x["target"] or ""))

def _find_attributes(captions: list[str], object_terms: list[str]) -> list[dict]:
    records = []
    for cap_idx, caption in enumerate(captions):
        words = tokens(caption)
        mentions = phrase_mentions(caption, object_terms)
        for idx, word in enumerate(words):
            if word not in ATTRIBUTE_TERMS:
                continue
            after = [m for m in mentions if m["start"] > idx and m["start"] - idx <= 4]
            before = [m for m in mentions if m["end"] <= idx and idx - m["end"] <= 3]
            obj = after[0]["concept"] if after else (before[-1]["concept"] if before else None)
            if obj:
                records.append({"attribute": word, "object": obj, "caption_index": cap_idx})
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["attribute"], row["object"])].append(row["caption_index"])
    out = []
    for (attr, obj), support_ids in grouped.items():
        out.append({"attribute": attr, "object": obj, "phrase": f"{attr} {obj}",
                    "support": len(set(support_ids)), "support_ids": sorted(set(support_ids))})
    return sorted(out, key=lambda x: (-x["support"], x["attribute"], x["object"]))

def _contains_sequence(words: list[str], phrase: str) -> bool:
    needle = tokens(phrase)
    return bool(needle) and any(words[i:i+len(needle)] == needle for i in range(len(words)-len(needle)+1))

def _find_relations(captions: list[str], object_terms: list[str], min_support: int) -> list[dict]:
    relation_rows = []
    relation_phrases = sorted(RELATIONS, key=lambda x: -len(x[0].split()))
    for cap_idx, caption in enumerate(captions):
        words = tokens(caption)
        mentions = phrase_mentions(caption, object_terms)
        for i, left in enumerate(mentions):
            for right in mentions[i + 1:]:
                if left["end"] >= right["start"]:
                    continue
                middle = words[left["end"]:right["start"]]
                relation = next((normalized for raw, normalized in relation_phrases
                                 if _contains_sequence(middle, raw)), None)
                if relation:
                    relation_rows.append({"subject": left["concept"], "relation": relation,
                                          "object": right["concept"], "caption_index": cap_idx})
    grouped = defaultdict(list)
    for row in relation_rows:
        grouped[(row["subject"], row["relation"], row["object"])].append(row["caption_index"])
    out = []
    for (subject, relation, obj), support_ids in grouped.items():
        support = len(set(support_ids))
        if support >= min_support:
            out.append({"subject": subject, "relation": relation, "object": obj,
                        "support": support, "support_ids": sorted(set(support_ids))})
    return sorted(out, key=lambda x: (-x["support"], x["subject"], x["relation"], x["object"]))

def _option(role, caption, concepts, score, assigned_role=None, fallback=False, reason=None, main=None):
    return {"role": role, "assigned_role": assigned_role or role, "caption": caption,
            "concepts": list(dict.fromkeys(concepts)), "score": float(max(0.0, min(1.5, score))),
            "fallback": bool(fallback), "fallback_reason": reason, "main_concept": main}

def _raw_fallback_options(role, captions, selected, limit=5):
    ranked = []
    for i, cap in enumerate(captions):
        overlap = max([lexical_overlap(cap, x["caption"]) for x in selected] or [0.0])
        ranked.append((overlap, i, cap))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return [_option(role, cap, [], 0.12 - 0.02*overlap,
                    assigned_role="raw_caption_fallback", fallback=True,
                    reason="no_reliable_role_or_residual_concept")
            for overlap, _, cap in ranked[:limit]]

def _make_role_options(role, state, data, cfg):
    pool, raw = data["pool"], data["captions"]
    main, used = state.get("main"), state["used"]
    objects = sorted([x for x in pool.values() if x["type"] == "object"],
                     key=lambda x: (-x["confidence"], -x["raw_support"], x["concept"]))
    scenes = sorted([x for x in pool.values() if x["type"] == "scene"],
                    key=lambda x: (-x["confidence"], -x["raw_support"], x["concept"]))
    options = []

    if role == "C1":
        mains = [x for x in objects if x["raw_support"] > 0]
        actions = _find_actions(raw, data["object_terms"])
        action_support = defaultdict(int)
        for action_row in actions:
            action_support[norm_phrase(action_row["subject"])] = max(
                action_support[norm_phrase(action_row["subject"])], action_row["support"])
        # Main-object selection is anchored in raw-caption subject position and
        # action-subject evidence. Novel F5/CoCu candidates must not hijack C1.
        mains.sort(key=lambda x: (-(0.45*x["first_support"]
                                    + 0.30*(action_support[norm_phrase(x["concept"])] / max(1, len(raw)))
                                    + 0.15*x["raw_support"]
                                    + 0.10*x["confidence"]), x["concept"]))
        for item in mains[:cfg["max_options"]]:
            action_rows = [a for a in actions if norm_phrase(a["subject"]) == norm_phrase(item["concept"])]
            action = action_rows[0] if action_rows and action_rows[0]["support"] >= cfg["min_action_support"] else None
            main_score = (0.45*item["first_support"]
                          + 0.30*(action_support[norm_phrase(item["concept"])] / max(1, len(raw)))
                          + 0.15*item["raw_support"] + 0.10*item["confidence"])
            if action:
                target = f" {article(action['target'])}" if action.get("target") else ""
                sentence = f"A photo of {article(item['concept'])} {action['action']}{target}."
                concepts = [item["concept"]] + ([action["target"]] if action.get("target") else [])
                score = main_score + 0.05*(action["support"]/max(1, len(raw)))
                options.append(_option(role, sentence, concepts, score, main=item["concept"]))
            sentence = f"A photo of {article(item['concept'])}."
            options.append(_option(role, sentence, [item["concept"]], main_score, main=item["concept"]))
        return sorted(options, key=lambda x: (-x["score"], x["caption"]))[:cfg["max_options"]] or _raw_fallback_options(role, raw, state["selected"])

    if role == "C2":
        attrs = _find_attributes(raw, data["object_terms"])
        if main:
            matched = [a for a in attrs if norm_phrase(a["object"]) == norm_phrase(main)]
            attrs = matched or attrs
        for a in attrs[:cfg["max_options"]]:
            sentence = f"A photo of {article(main)} with {article(a['phrase'])}." if main else f"A photo showing {article(a['phrase'])}."
            options.append(_option(role, sentence, [a["phrase"]], min(1.0, 0.45 + 0.12*a["support"]), main=main))
        if not options:
            residual = [x for x in objects if norm_phrase(x["concept"]) != norm_phrase(main or "") and norm_phrase(x["concept"]) not in used]
            for x in residual[:cfg["max_options"]]:
                options.append(_option(role, f"A photo showing {article(x['concept'])}.", [x["concept"]],
                    0.45*x["confidence"] + 0.30*x["raw_support"], assigned_role="secondary_object_fallback",
                    fallback=True, reason="no_reliable_attribute", main=main))
        return options or _raw_fallback_options(role, raw, state["selected"])

    if role == "C3":
        residual = [x for x in objects if norm_phrase(x["concept"]) != norm_phrase(main or "") and norm_phrase(x["concept"]) not in used]
        for x in residual[:cfg["max_options"]]:
            options.append(_option(role, f"A photo showing {article(x['concept'])}.", [x["concept"]],
                0.55*x["confidence"] + 0.45*x["raw_support"], assigned_role="local_object", main=main))
        return options or _raw_fallback_options(role, raw, state["selected"])

    if role == "C4":
        relations = _find_relations(raw, data["object_terms"], cfg["min_relation_support"])
        for r in relations[:cfg["max_options"]]:
            sentence = f"A photo of {article(r['subject'])} {r['relation']} {article(r['object'])}."
            options.append(_option(role, sentence, [r["subject"], r["object"]],
                min(1.0, 0.45 + 0.12*r["support"]), assigned_role="spatial_relation", main=main))
        if not options:
            residual = [x for x in objects if norm_phrase(x["concept"]) != norm_phrase(main or "") and norm_phrase(x["concept"]) not in used]
            for x in residual[:cfg["max_options"]]:
                options.append(_option(role, f"A photo showing {article(x['concept'])}.", [x["concept"]],
                    0.35*x["confidence"] + 0.25*x["raw_support"], assigned_role="residual_object_fallback",
                    fallback=True, reason="insufficient_spatial_relation_evidence", main=main))
        return options or _raw_fallback_options(role, raw, state["selected"])

    if role == "C5":
        for x in scenes[:cfg["max_options"]]:
            options.append(_option(role, f"A photo of {article(x['concept'])}.", [x["concept"]],
                0.60*x["confidence"] + 0.40*x["raw_support"], assigned_role="scene_context", main=main))
        if not options:
            residual = [x for x in objects if norm_phrase(x["concept"]) != norm_phrase(main or "") and norm_phrase(x["concept"]) not in used]
            for x in residual[:cfg["max_options"]]:
                options.append(_option(role, f"A photo showing {article(x['concept'])}.", [x["concept"]],
                    0.40*x["confidence"] + 0.30*x["raw_support"], assigned_role="residual_object_fallback",
                    fallback=True, reason="no_reliable_scene_concept", main=main))
        return options or _raw_fallback_options(role, raw, state["selected"])
    raise ValueError(f"Unknown role: {role}")

def build_candidate_pool(raw_captions, source_rows, gallery_terms=None):
    """Merge sources with source-local rank normalization and raw-caption support."""
    gallery_terms = gallery_terms or []
    source_terms = [str(r.get("concept", "")).strip() for rows in source_rows.values() for r in rows]
    terms = list(dict.fromkeys([x for x in chain(gallery_terms, source_terms) if x]))
    raw_mentions = [m for cap in raw_captions for m in phrase_mentions(cap, terms)]
    # Keep only candidates supported by a source or actually mentioned in the raw
    # captions. The full gallery is a matcher vocabulary, not a per-image candidate set.
    canonical = {}
    for term in source_terms:
        key = norm_phrase(term)
        if key:
            canonical.setdefault(key, term)
    for mention in raw_mentions:
        key = norm_phrase(mention["concept"])
        if key:
            canonical.setdefault(key, mention["concept"])
    terms = list(canonical.values())

    raw_count = defaultdict(set)
    first_count = defaultdict(set)
    for idx, cap in enumerate(raw_captions):
        mentions = phrase_mentions(cap, terms)
        for m in mentions:
            raw_count[norm_phrase(m["concept"])].add(idx)
        # The first recognized object mention is a weak subject-position cue.
        object_mentions = [m for m in mentions
                           if classify_concept(m["concept"]) == "object"]
        if object_mentions:
            first_count[norm_phrase(object_mentions[0]["concept"])].add(idx)

    source_rank, source_meta = defaultdict(dict), defaultdict(list)
    for source, rows in source_rows.items():
        ranks = rank_normalize(rows)
        for row in rows:
            concept = str(row.get("concept", "")).strip()
            key = norm_phrase(concept)
            if not key:
                continue
            source_rank[key][source] = ranks.get(concept, 0.5)
            source_meta[key].append({
                "source": source, "score": row.get("score", row.get("target_score")),
                "best_head": row.get("best_head"),
                "grounding_score": row.get("grounding_score", row.get("p1_score")),
            })

    pool, n_sources = {}, max(1, len(source_rows))
    for key, concept in canonical.items():
        ranks = source_rank.get(key, {})
        if ranks:
            weights = {s: SOURCE_WEIGHTS.get(s, 0.7) for s in ranks}
            rank_score = sum(ranks[s]*weights[s] for s in ranks) / sum(weights.values())
        else:
            rank_score = 0.5
        agreement = len(ranks) / n_sources
        support = len(raw_count.get(key, set())) / max(1, len(raw_captions))
        meta = source_meta.get(key, [])
        grounding_values = []
        for m in meta:
            try:
                g = float(m["grounding_score"])
                if 0.0 <= g <= 1.0:
                    grounding_values.append(g)
            except (TypeError, ValueError):
                pass
        if grounding_values:
            grounding = sum(grounding_values) / len(grounding_values)
            confidence = 0.52*rank_score + 0.18*agreement + 0.15*support + 0.15*grounding
        else:
            grounding = None
            confidence = 0.62*rank_score + 0.20*agreement + 0.18*support
        pool[key] = {
            "concept": concept, "type": classify_concept(concept),
            "confidence": max(0.0, min(1.0, confidence)),
            "source_rank": rank_score, "source_agreement": agreement,
            "raw_support": support, "raw_support_count": len(raw_count.get(key, set())),
            "first_support": len(first_count.get(key, set())) / max(1, len(raw_captions)),
            "sources": sorted(ranks), "source_evidence": meta, "grounding_score": grounding,
        }
    object_terms = [x["concept"] for x in pool.values() if x["type"] == "object"]
    return pool, object_terms

def compose_one_image(image_id, raw_captions, source_rows, gallery_terms=None, config=None):
    cfg = {"beam_width": 32, "max_options": 5, "min_action_support": 1,
           "min_relation_support": 2, "coverage_weight": 0.12, "redundancy_weight": 0.18}
    if config:
        cfg.update(config)
    pool, object_terms = build_candidate_pool(raw_captions, source_rows, gallery_terms)
    scene_terms = [x["concept"] for x in pool.values() if x["type"] == "scene"]
    data = {"pool": pool, "captions": raw_captions, "object_terms": list(dict.fromkeys(object_terms + scene_terms))}
    beam = [{"selected": [], "used": set(), "score": 0.0, "main": None}]
    for role in ("C1", "C2", "C3", "C4", "C5"):
        expanded = []
        for state in beam:
            for opt in _make_role_options(role, state, data, cfg):
                overlap = max([lexical_overlap(opt["caption"], x["caption"]) for x in state["selected"]] or [0.0])
                new_concepts = {norm_phrase(x) for x in opt["concepts"]} - state["used"]
                repeats = sum(1 for x in opt["concepts"] if norm_phrase(x) in state["used"])
                score = state["score"] + opt["score"] + cfg["coverage_weight"]*len(new_concepts) \
                        - cfg["redundancy_weight"]*overlap - 0.04*repeats
                expanded.append({
                    "selected": state["selected"] + [opt],
                    "used": state["used"] | {norm_phrase(x) for x in opt["concepts"]},
                    "score": score, "main": state["main"] or opt.get("main_concept"),
                })
        expanded.sort(key=lambda x: (-x["score"], tuple(y["caption"] for y in x["selected"])))
        beam = expanded[:cfg["beam_width"]]
    best = beam[0]
    captions = [x["caption"] for x in best["selected"]]
    if len(captions) != 5:
        raise AssertionError(f"Expected exactly five captions, got {len(captions)}")
    return {
        "image_id": int(image_id), "raw_captions": list(raw_captions),
        "captions": captions, "roles": best["selected"],
        "composition_score": float(best["score"]), "candidate_pool": list(pool.values()),
        "diagnostics": {
            "candidate_count": len(pool),
            "object_count": sum(x["type"] == "object" for x in pool.values()),
            "scene_count": sum(x["type"] == "scene" for x in pool.values()),
            "action_evidence": _find_actions(raw_captions, data["object_terms"]),
            "attribute_evidence": _find_attributes(raw_captions, data["object_terms"]),
            "relation_evidence": _find_relations(raw_captions, data["object_terms"], cfg["min_relation_support"]),
        },
    }
