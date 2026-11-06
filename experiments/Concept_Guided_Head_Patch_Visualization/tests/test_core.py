import unittest

import torch

from experiments.Concept_Guided_Head_Patch_Visualization.core import (
    concept_patch_cosine,
    extract_head_patch_relations,
    propagate_concept_scores,
    retrieve_head_concepts,
)


class TestHeadPatchCore(unittest.TestCase):
    def test_p0_p1_p2_shapes_and_normalization(self):
        torch.manual_seed(7)
        heads, dim, patches, globals_ = 2, 8, 9, 2
        qkv = torch.randn(1, globals_ + patches, 3 * dim)
        patch_tokens = torch.randn(1, patches, dim)
        out = extract_head_patch_relations(
            qkv, patch_tokens, num_heads=heads, num_global_tokens=globals_, query_index=4
        )
        self.assertEqual(tuple(out["heads"].shape), (heads, dim))
        self.assertEqual(tuple(out["p0_cls_to_patch"].shape), (heads, patches))
        self.assertEqual(tuple(out["p1_affinity"].shape), (heads, patches, patches))
        self.assertEqual(tuple(out["p2_mean_feature_affinity"].shape), (heads, patches))
        self.assertTrue(torch.allclose(out["p0_cls_to_patch"].sum(-1), torch.ones(heads), atol=1e-6))
        self.assertTrue(torch.allclose(out["p1_affinity"].sum(-1), torch.ones(heads, patches), atol=1e-6))
        self.assertTrue(torch.allclose(out["p1_mean_incoming"].sum(-1), torch.ones(heads), atol=1e-6))

    def test_concept_patch_and_relation_propagation(self):
        patches = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        concepts = torch.tensor([[1., 0.], [0., 1.]])
        scores = concept_patch_cosine(patches, concepts)
        self.assertEqual(tuple(scores.shape), (3, 2))
        affinity = torch.eye(3).unsqueeze(0)
        propagated = propagate_concept_scores(affinity, scores)
        self.assertTrue(torch.allclose(propagated[0], scores))

    def test_head_gallery_retrieval(self):
        heads = torch.tensor([[1., 0.], [0., 1.]])
        gallery_features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        scores, rows = retrieve_head_concepts(
            heads, gallery_features, ["cat", "grass", "car"], topk=2
        )
        self.assertEqual(tuple(scores.shape), (2, 3))
        self.assertEqual(rows[0][0]["concept"], "cat")
        self.assertEqual(rows[1][0]["concept"], "grass")


if __name__ == "__main__":
    unittest.main()
