#!/usr/bin/env python3
"""Transparent lexical multi-label classifier for ECSD candidate roles.

This is a diagnostic heuristic, not a semantic ground-truth annotator.
"""
import re

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

ENTITY_TERMS = set("""
person people man woman boy girl child children baby toddler adult
dog cat horse cow sheep goat bird duck chicken elephant bear zebra giraffe
lion tiger deer rabbit squirrel monkey animal
bicycle bike motorcycle scooter car truck bus train airplane plane boat ship
surfboard skateboard snowboard skis racket tennis racket baseball bat
ball frisbee kite umbrella backpack bag suitcase handbag
chair table bench couch sofa bed desk cabinet shelf bottle cup glass bowl
plate fork knife spoon pot pan
laptop computer keyboard mouse monitor phone cellphone camera television tv
book newspaper magazine clock watch vase flower plant tree branch leaf
sign traffic light hydrant mailbox fire hydrant stop sign
building house home apartment store shop restaurant stadium court field
street road sidewalk bridge fence wall door window roof
""".split())

ATTRIBUTE_TERMS = set("""
red blue green yellow black white brown orange pink purple gray grey
silver golden gold colorful colour colored coloured dark light bright
small little tiny large big huge tall short long wide narrow
young old happy sad smiling cute beautiful dirty clean wet dry
wooden metal plastic glass ceramic striped spotted patterned
round square flat thick thin fluffy furry shiny smooth rough
wearing dressed striped polka dotted
""".split())

ACTION_TERMS = set("""
stand stands standing stood sit sits sitting seated
lie lies lying lay ride rides riding rode
walk walks walking walked run runs running ran
play plays playing played hold holds holding held
carry carries carrying carried eat eats eating ate
drink drinks drinking drank look looks looking looked
watch watches watching watched chase chases chasing chased
throw throws throwing threw catch catches catching caught
jump jumps jumping jumped fly flies flying flew
swim swims swimming swam surf surfs surfing ski skis skiing
skate skates skating snowboard snowboards snowboarding
drive drives driving drove park parks parking parked
sleep sleeps sleeping slept talk talks talking smile smiles
cut cuts cutting cook cooks cooking wash washes washing
brush brushes brushing open opens opening opened close closes closing closed
use uses using used type types typing read reads reading
take takes taking took make makes making made feed feeds feeding
wear wears wearing worn wore push pushes pushing pulled pull pulls pulling
kick kicks kicking kicked work works working worked point points pointing
lean leans leaning bend bends bending
""".split())

RELATION_TERMS = set("""
in on at to from into onto with without over under above below
near beside next behind between around through across along inside outside
front left right top bottom middle beside among against toward towards
""".split())

CONTEXT_TERMS = set("""
grass field street road sidewalk beach water ocean sea sky snow
mountain hill park room kitchen bathroom bedroom table forest woods
city building house court stadium store shop restaurant farm
sand pavement floor wall ground landscape river lake garden
indoor outdoor background countryside playground parking lot
""".split())

CONNECTOR_TERMS = set("""
and or but while which who that with in on at to from of for by
into onto over under near beside behind between around through across along
""".split())

IRREGULAR = {
    "men": "man", "women": "woman", "children": "child", "people": "person",
    "bicycles": "bicycle", "bikes": "bike", "dogs": "dog", "cats": "cat",
    "horses": "horse", "riding": "ride", "rides": "ride", "rode": "ride",
    "lying": "lie", "lies": "lie", "lay": "lie", "flying": "fly",
    "flies": "fly", "flew": "fly", "running": "run", "ran": "run",
    "sitting": "sit", "sits": "sit", "stood": "stand", "standing": "stand",
    "stands": "stand", "swimming": "swim", "swam": "swim",
    "wearing": "wear", "worn": "wear", "wore": "wear",
    "holding": "hold", "held": "hold", "eating": "eat", "ate": "eat",
    "drinking": "drink", "drank": "drink", "chasing": "chase",
    "chased": "chase", "throwing": "throw", "threw": "throw",
    "catching": "catch", "caught": "catch", "playing": "play",
    "played": "play", "walking": "walk", "walked": "walk",
    "looking": "look", "looked": "look", "watching": "watch",
    "watched": "watch", "jumping": "jump", "jumped": "jump",
    "carrying": "carry", "carried": "carry", "cutting": "cut",
    "cooked": "cook", "driving": "drive", "drove": "drive",
    "parked": "park", "parking": "park",
}

def normalize_token(token):
    token = token.lower().strip("-'")
    if token in IRREGULAR:
        return IRREGULAR[token]
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
    return {
        normalize_token(x) for x in WORD_RE.findall(str(text).lower())
        if normalize_token(x) and normalize_token(x) not in STOPWORDS
        and not normalize_token(x).isdigit()
    }

def classify_semantic_roles(text):
    raw_tokens = {x.lower() for x in WORD_RE.findall(str(text).lower())}
    raw = raw_tokens | {normalize_token(x) for x in raw_tokens}
    lexicons = {
        "Entity": ENTITY_TERMS,
        "Attribute": ATTRIBUTE_TERMS,
        "Action": ACTION_TERMS,
        "Relation": RELATION_TERMS,
        "Context": CONTEXT_TERMS,
    }
    rules = {name: sorted(raw & terms) for name, terms in lexicons.items()}
    categories = [name for name, hits in rules.items() if hits]
    if not categories:
        categories = ["Unknown"]
    return {"categories": categories, "rules": rules, "unknown": categories == ["Unknown"]}

def quality_flags(text, candidate_type="candidate", filter_config=None):
    """Return transparent surface-form warnings; never treat them as truth labels."""
    words = [normalize_token(x) for x in WORD_RE.findall(str(text).lower())]
    facts = content_tokens(text)
    flags = []
    filter_config = filter_config or {}
    if not words or not facts:
        flags.append("no_content_tokens")
    if words and words[0] in CONNECTOR_TERMS:
        flags.append("leading_connector")
    if words and words[-1] in CONNECTOR_TERMS:
        flags.append("dangling_connector_or_preposition")
    minimums = filter_config.get("min_content_tokens", {})
    if isinstance(minimums, dict):
        minimum = int(minimums.get(candidate_type, 1 if candidate_type == "noun_phrase" else 2))
    else:
        minimum = int(minimums) if minimums else (1 if candidate_type == "noun_phrase" else 2)
    if len(facts) < minimum:
        flags.append("too_few_content_tokens")
    if len(words) > int(filter_config.get("max_words", 36)):
        flags.append("overlong_candidate")
    if str(text).count("(") != str(text).count(")"):
        flags.append("unbalanced_parentheses")
    return flags

def normalize_text(text):
    return " ".join(normalize_token(x) for x in WORD_RE.findall(str(text).lower())
                    if normalize_token(x) not in STOPWORDS)
