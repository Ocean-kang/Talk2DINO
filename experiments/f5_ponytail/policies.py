"""Small deterministic policies shared by the builder and tests."""
from itertools import permutations

# threshold, assignment, negatives, lambda
VARIANTS = {
    "f5_control": (False, "clip", "lower_half", 0.3),
    "threshold": (True, "clip", "lower_half", 0.3),
    "head": (True, "head", "lower_half", 0.3),
    "balanced": (True, "balanced", "lower_half", 0.3),
    "cf_l03": (True, "balanced", "filtered", 0.3),
    "cf_l00": (True, "balanced", "filtered", 0.0),
    "cf_l01": (True, "balanced", "filtered", 0.1),
    "cf_l06": (True, "balanced", "filtered", 0.6),
}
SEEDS = (123, 124, 125, 126, 127)


def assign(costs, balanced):
    if not costs:
        return ()
    slots = range(len(costs[0]))
    if not balanced:
        return tuple(min(slots, key=row.__getitem__) for row in costs)
    # At most 3 concepts and 5 captions, so enumerate at most 60 assignments.
    return min(permutations(slots, len(costs)),
               key=lambda indices: (sum(row[i] for row, i in zip(costs, indices)), indices))


def select(rows, coverage, policy, floor, margin, max_concepts):
    threshold, _, strategy, lam = policy
    result = []
    for row in rows:
        target = row["target_score"]
        gate = max(floor, coverage[row["best_head"]] + margin)
        if threshold and target < gate:
            continue
        peers = row[strategy]
        penalty = sum(score for _, score in peers) / len(peers) if peers else 0.0
        result.append(dict(row, gate=gate if threshold else None, score=target - lam * penalty,
                           negative_mean=penalty if peers else None,
                           negative_ids=[iid for iid, _ in peers]))
    return sorted(result, key=lambda r: (-r["score"], r["concept"]))[:max_concepts]
