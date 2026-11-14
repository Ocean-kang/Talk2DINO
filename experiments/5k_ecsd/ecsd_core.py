#!/usr/bin/env python3
"""Deterministic, evidence-constrained candidate construction for ECSD-naive."""
import math
import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+")

STOPWORDS = set("""
a an the this that these those some any each every another all both either
is are was were be been being am has have had do does did doing
and or but while where which who whose whom what when
of in on at to from for by as into onto with without over under above below
near beside next behind between around through across along inside outside
up down off out about after before during
it its they them their he she his her we our you your i me my
there here one two three photo picture image scene view showing shows show
""".split())

ARTICLES = {"a", "an", "the"}

ACTION_WORDS = set("""
stand stands standing stood sit sits sitting seated
lie lies lying lay
ride rides riding rode
walk walks walking walked run runs running ran
play plays playing played
hold holds holding held carry carries carrying carried
eat eats eating ate drink drinks drinking drank
look looks looking looked watch watches watching watched
chase chases chasing chased
throw throws throwing threw catch catches catching caught
jump jumps jumping jumped fly flies flying flew
swim swims swimming swam
surf surfs surfing ski skis skiing skate skates skating
snowboard snowboards snowboarding
drive drives driving drove park parks parking parked
sleep sleeps sleeping slept
talk talks talking talked smile smiles smiling
cut cuts cutting cook cooks cooking
wash washes washing brush brushes brushing
open opens opening opened close closes closing closed
use uses using used type types typing read reads reading
take takes taking took make makes making made
feed feeds feeding
wear wears wearing worn
push pushes pushing pulled pull pulls pulling
kick kicks kicking kicked
work works working worked
point points pointing
lean leans leaning
bend bends bending
""".split())

ATTRIBUTE_WORDS = set("""
red blue green yellow black white brown orange pink purple gray grey
silver golden gold colorful colour colored coloured dark light bright
small little tiny large big huge tall short long wide narrow
young old happy smiling cute beautiful dirty clean wet dry
wooden metal plastic glass ceramic striped spotted patterned
round square flat thick thin fluffy furry shiny
""".split())

RELATION_WORDS = set("""
in on at to from into onto with without over under above below
near beside next behind between around through across along inside outside
front left right top bottom middle
""".split())

CONTEXT_WORDS = set("""
grass field street road sidewalk beach water ocean sea sky snow
mountain hill park room kitchen bathroom bedroom table forest woods
city building house court stadium store shop restaurant farm
sand pavement floor wall ground landscape
""".split())

NP_STOP = (STOPWORDS - {"and", "or"}) | ACTION_WORDS | {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "in", "on", "at", "to", "from", "for", "by", "as",
    "into", "onto", "with", "without", "over", "under",
    "above", "below", "near", "beside", "behind", "between",
    "around", "through", "across", "along", "inside", "outside",
    "while", "where", "which", "who", "that", "but"
}


def _normalise_token(token):
    token = token.lower().strip("-'")
    irregular = {
        "riding": "ride", "rides": "ride", "rode": "ride",
        "lying": "lie", "lies": "lie", "lay": "lie",
        "flying": "fly", "flies": "fly", "flew": "fly",
        "running": "run", "ran": "run",
        "sitting": "sit", "sits": "sit", "stood": "stand",
        "standing": "stand", "stands": "stand",
        "swimming": "swim", "swam": "swim",
        "wearing": "wear", "worn": "wear", "wore": "wear",
        "holding": "hold", "held": "hold",
        "eating": "eat", "ate": "eat",
        "drinking": "drink", "drank": "drink",
        "chasing": "chase", "chased": "chase",
        "throwing": "throw", "threw": "throw",
        "catching": "catch", "caught": "catch",
        "playing": "play", "played": "play",
        "walking": "walk", "walked": "walk",
        "looking": "look", "looked": "look",
        "watching": "watch", "watched": "watch",
        "jumping": "jump", "jumped": "jump",
        "carrying": "carry", "carried": "carry",
        "cutting": "cut", "cooked": "cook",
        "driving": "drive", "drove": "drive",
        "parked": "park", "parking": "park",
    }
    if token in irregular:
        return irregular[token]
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        root = token[:-3]
        if len(root) > 2 and root[-1] == root[-2]:
            root = root[:-1]
        if root in {"rid", "mak", "us", "tak", "giv", "lov"}:
            root += "e"
        return root
    if len(token) > 4 and token.endswith("ed"):
        root = token[:-2]
        if root.endswith("i"):
            root = root[:-1] + "y"
        return root
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text):
    result = set()
    for raw in WORD_RE.findall(text.lower()):
        token = _normalise_token(raw)
        if token and token not in STOPWORDS and not token.isdigit():
            result.add(token)
    return result


def _roles(text, candidate_type):
    raw = set(x.lower() for x in WORD_RE.findall(text))
    if candidate_type == "noun_phrase":
        if raw & ATTRIBUTE_WORDS:
            return "attribute"
        return "entity"
    if raw & ACTION_WORDS:
        return "action"
    if raw & RELATION_WORDS:
        return "relation"
    if raw & CONTEXT_WORDS:
        return "context"
    return "main"


def _sentence(text):
    text = re.sub(r"\s+", " ", text.strip(" \t\r\n,;:"))
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _quality(text, candidate_type):
    base = {"original": 1.0, "clause": 0.82, "noun_phrase": 0.74}[candidate_type]
    n = len(WORD_RE.findall(text))
    if n < 5:
        base *= 0.72
    elif n > 28:
        base *= 0.86
    return round(base, 4)


def build_candidates(captions):
    """Build source-grounded candidates; all semantic spans come from inputs."""
    candidates = []
    generated_by_text = {}

    def add(text, caption_index, start, end, candidate_type, deduplicate):
        text = _sentence(text)
        if not text:
            return
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if not key:
            return
        if deduplicate and key in generated_by_text:
            existing = generated_by_text[key]
            existing["source_caption_ids"].append(caption_index)
            existing["source_spans"].append({
                "caption_index": caption_index, "start": int(start), "end": int(end)
            })
            return

        facts = sorted(content_tokens(text))
        if not facts:
            return
        item = {
            "candidate_id": "{}:{}:{}:{}".format(
                caption_index, candidate_type, start, end
            ),
            "text": text,
            "candidate_type": candidate_type,
            "source_caption_ids": [caption_index],
            "source_spans": [{
                "caption_index": caption_index, "start": int(start), "end": int(end)
            }],
            "facts": facts,
            "role": _roles(text, candidate_type),
            "quality": _quality(text, candidate_type),
        }
        candidates.append(item)
        if deduplicate:
            generated_by_text[key] = item

    for caption_index, caption in enumerate(captions):
        caption = caption.strip()
        add(caption, caption_index, 0, len(caption), "original", False)

        # Only explicit punctuation boundaries are split. We do not split noun
        # coordination such as "a man and a woman".
        for match in re.finditer(r"[^,;:.!?]+", caption):
            fragment = match.group(0).strip()
            if len(content_tokens(fragment)) >= 4:
                first = WORD_RE.search(fragment)
                if first and first.group(0).lower() not in {
                    "and", "or", "but", "while", "which", "who", "that"
                }:
                    start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
                    end = start + len(fragment)
                    add(fragment, caption_index, start, end, "clause", True)

        # Lightweight noun-phrase extraction. Stop at common verbs,
        # prepositions, and clause markers. No parser/model is required.
        tokens = list(WORD_RE.finditer(caption))
        for idx, match in enumerate(tokens):
            if match.group(0).lower() not in ARTICLES:
                continue
            end_idx = idx + 1
            while end_idx < len(tokens) and (end_idx - idx) < 9:
                word = tokens[end_idx].group(0).lower()
                if word in NP_STOP:
                    break
                end_idx += 1
            if end_idx <= idx + 1:
                continue
            last = tokens[end_idx - 1]
            phrase = caption[match.start():last.end()].strip()
            phrase_facts = content_tokens(phrase)
            if len(phrase_facts) < 2:
                continue
            lower_phrase = phrase[0].lower() + phrase[1:]
            candidate_text = "A photo shows {}.".format(lower_phrase)
            add(
                candidate_text, caption_index, match.start(), last.end(),
                "noun_phrase", True
            )

    return candidates


def build_fact_weights(captions):
    """Inverse-frequency weights over unique content facts in source captions."""
    frequency = Counter()
    for caption in captions:
        frequency.update(content_tokens(caption))
    n = max(1, len(captions))
    weights = {
        token: 1.0 + math.log((n + 1.0) / (count + 1.0))
        for token, count in frequency.items()
    }
    return frequency, weights


def weighted_jaccard(a, b, weights):
    a, b = set(a), set(b)
    union = a | b
    if not union:
        return 0.0
    numerator = sum(weights.get(token, 1.0) for token in (a & b))
    denominator = sum(weights.get(token, 1.0) for token in union)
    return numerator / denominator if denominator else 0.0
