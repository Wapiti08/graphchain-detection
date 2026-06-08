from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path


def _build_stage_labels(**kwargs):
    from gchain.train.modeling import build_stage_labels

    return build_stage_labels(**kwargs)


class TestStageSupervisionLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch not installed")
        from gchain.eval.attack_reconstruct import ioc_type_to_stage_idx  # noqa: F401

        cls._repo_root = Path(__file__).resolve().parents[1]

    def _stream(self, *, rule_types, y_rule_high):
        import torch

        n = len(rule_types)
        return SimpleNamespace(
            src=torch.zeros(n, dtype=torch.long),
            rule_ioc_type=tuple(rule_types),
            y_rule_high=torch.tensor(y_rule_high, dtype=torch.long),
            y_rule=torch.tensor(y_rule_high, dtype=torch.long),
            ioc_type=tuple("" for _ in range(n)),
        )

    def test_rule_high_only_labeled_edges(self) -> None:
        from gchain.eval.attack_reconstruct import ioc_type_to_stage_idx, load_ioc_type_to_stage

        st = self._stream(
            rule_types=("execution", "", "attack_ip"),
            y_rule_high=[1, 0, 1],
        )
        out = _build_stage_labels(
            streams={"sc9": st},
            repo_root=self._repo_root,
            ioc_type_to_stage_idx=ioc_type_to_stage_idx,
            load_stage_map=load_ioc_type_to_stage,
            stage_supervision="rule_high",
        )
        labels = out["sc9"]
        self.assertGreater(int(labels[0].item()), 0)
        self.assertEqual(int(labels[1].item()), 0)
        self.assertEqual(int(labels[2].item()), 0)

    def test_resolve_falls_back_rule_high_to_rule(self) -> None:
        import torch

        from gchain.eval.attack_reconstruct import ioc_type_to_stage_idx, load_ioc_type_to_stage
        from gchain.train.modeling import resolve_stage_supervision

        st = SimpleNamespace(
            src=torch.zeros(2, dtype=torch.long),
            rule_ioc_type=("attack_ip", "execution"),
            y_rule_high=torch.tensor([1, 0], dtype=torch.long),
            y_rule=torch.tensor([1, 1], dtype=torch.long),
            ioc_type=("", ""),
        )
        mode, use_stage = resolve_stage_supervision(
            streams={"sc1": st},
            stage_supervision="rule_high",
            repo_root=self._repo_root,
            ioc_type_to_stage_idx=ioc_type_to_stage_idx,
            load_stage_map=load_ioc_type_to_stage,
        )
        self.assertEqual(mode, "rule")
        self.assertTrue(use_stage)


if __name__ == "__main__":
    unittest.main()
