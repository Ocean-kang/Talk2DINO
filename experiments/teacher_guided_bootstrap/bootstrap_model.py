"""Teacher-guided caption-to-head selection for Talk2DINO.

This module intentionally changes only the hard attention-head selection used by
ProjectionLayer(alignment_strategy="max_score") during training. The Teacher is
frozen and is not registered as a Student submodule, so Student checkpoints keep
exactly the original ProjectionLayer state-dict structure.
"""

from __future__ import annotations

import weakref

import torch
import torch.nn.functional as F

from src.model import ProjectionLayer


class TeacherGuidedProjectionLayer(ProjectionLayer):
    """ProjectionLayer with Teacher-guided head selection during training only.

    For one caption and its H DINO attention-head features, the original model
    computes Student head scores and takes a hard argmax. Here the selection score
    becomes

        score = (1 - alpha) * student_score + alpha * teacher_score

    where both scores use the same per-head softmax already present in the
    original Talk2DINO max_score implementation. After selecting the head, the
    batch similarity matrix is still computed only by the Student and the normal
    Talk2DINO InfoNCE loss is used unchanged.
    """

    def attach_teacher(self, teacher: ProjectionLayer, selection_alpha: float) -> None:
        if not 0.0 <= selection_alpha <= 1.0:
            raise ValueError(f"selection_alpha must be in [0, 1], got {selection_alpha}")
        if teacher.alignment_strategy != "max_score":
            raise ValueError(
                "Teacher must use alignment_strategy='max_score' for this experiment"
            )
        if teacher.cosine != self.cosine:
            raise ValueError("Teacher and Student must use the same cosine setting")

        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        # Ponytail / side-car rule: keep Teacher outside Student's registered module
        # tree. This prevents teacher.* keys from leaking into the saved Student
        # checkpoint and keeps downstream evaluation fully compatible with the
        # repository's original ProjectionLayer.
        self._teacher_ref = weakref.ref(teacher)
        self.selection_alpha = float(selection_alpha)

    def _teacher(self) -> ProjectionLayer:
        teacher_ref = getattr(self, "_teacher_ref", None)
        teacher = None if teacher_ref is None else teacher_ref()
        if teacher is None:
            raise RuntimeError("Teacher is not attached or has been released")
        return teacher

    def train(self, mode: bool = True):
        # The Teacher is deliberately not a registered child module. Keep it in
        # eval mode explicitly whenever the Student training mode changes.
        super().train(mode)
        teacher_ref = getattr(self, "_teacher_ref", None)
        if teacher_ref is not None:
            teacher = teacher_ref()
            if teacher is not None:
                teacher.eval()
        return self

    def forward(
        self,
        visual_embedding,
        textual_embedding,
        ret_similarity_matrix=True,
        ret_embeds=False,
        self_attn_maps=None,
        cls=None,
        text_input_mask=None,
        return_index=False,
    ):
        alpha = float(getattr(self, "selection_alpha", 0.0))

        # alpha=0 is a literal pass-through to the repository implementation.
        # Evaluation/inference is also Student-only: Teacher affects the training
        # selection trajectory, not deployment-time prediction.
        if (not self.training) or alpha == 0.0:
            return super().forward(
                visual_embedding,
                textual_embedding,
                ret_similarity_matrix=ret_similarity_matrix,
                ret_embeds=ret_embeds,
                self_attn_maps=self_attn_maps,
                cls=cls,
                text_input_mask=text_input_mask,
                return_index=return_index,
            )

        if self.alignment_strategy != "max_score":
            raise RuntimeError(
                "Teacher-guided bootstrap is defined only for alignment_strategy='max_score'"
            )

        if self.weight_attn_heads is not None:
            visual_embedding = self.get_visual_embed(
                visual_embedding,
                self_attn_maps=self_attn_maps,
                cls=cls,
            )

        if visual_embedding.ndim != 3 or textual_embedding.ndim != 2:
            raise RuntimeError(
                "Teacher-guided selection expects disentangled DINO head features "
                "[B, H, D] and one CLIP caption embedding per sample [B, C]."
            )

        raw_text = textual_embedding
        student_text = self.project_clip_txt(raw_text)
        student_visual = visual_embedding

        if self.cosine:
            student_text = F.normalize(student_text, p=2, dim=-1)
            student_visual = F.normalize(student_visual, p=2, dim=-1)

        if ret_embeds:
            # Preserve ProjectionLayer's ret_embeds contract. Selection is not
            # needed when callers explicitly request embeddings.
            return student_text, student_visual

        teacher = self._teacher()
        with torch.no_grad():
            teacher_text = teacher.project_clip_txt(raw_text)
            teacher_visual = visual_embedding
            if teacher.cosine:
                teacher_text = F.normalize(teacher_text, p=2, dim=-1)
                teacher_visual = F.normalize(teacher_visual, p=2, dim=-1)

            teacher_head_scores = torch.einsum(
                "ik,ijk->ij", teacher_text, teacher_visual
            ).softmax(dim=-1)

        # Match the original max_score code path: dot product -> softmax over
        # heads -> hard argmax. Only the score used by argmax is blended.
        student_head_scores = torch.einsum(
            "ik,ijk->ij", student_text, student_visual
        ).softmax(dim=-1)
        bootstrap_scores = (
            (1.0 - alpha) * student_head_scores + alpha * teacher_head_scores
        )
        index = bootstrap_scores.argmax(dim=-1)
        gather_index = index.view(-1, 1, 1).expand(
            -1, 1, student_visual.shape[-1]
        )
        selected_visual = torch.gather(
            student_visual, 1, gather_index
        ).squeeze(1)

        # Original Talk2DINO batch similarity matrix. Teacher does not participate
        # here and therefore never appears in InfoNCE / gradients.
        scores = student_text @ selected_visual.transpose(1, 0)

        if not ret_similarity_matrix:
            diagonal = torch.eye(
                len(scores), dtype=torch.bool, device=scores.device
            )
            scores = scores[diagonal]

        if return_index:
            return scores, index
        return scores
