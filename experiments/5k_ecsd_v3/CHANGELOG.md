# ECSD v3 Changelog

## v3.0.0

- Reject full-caption Clause fragments.
- Deduplicate candidate surface strings per image and merge source provenance.
- Enforce Original > Clause > Noun Phrase representative-type priority.
- Introduce deterministic image/text-derived candidate IDs.
- Compute one shared anchor per image and reuse it for B1-B4.
- Determine manual audit Selected status by exact candidate ID.
- Add conservative annotation migration by image ID and normalized candidate text.
- Add atomic JSON/JSONL/CSV writes with timestamped backups on explicit overwrite.
- Validate candidate provenance, final JSON/Trace equality, and anchor invariance.
- Recompute CISD from final JSON using normalized CLIP embeddings and an incremental embedding store.
- Add final-output metric summaries and pairwise selection consistency.
- Keep the package separate from v2 and defer training/OVSS until A/B artifacts pass validation.
