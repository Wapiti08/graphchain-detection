"""Smoke tests for static GraphSAGE / RGCN baselines."""
from __future__ import annotations

import unittest


class TestStaticGNN(unittest.TestCase):
    def test_train_and_score_synthetic(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not installed")

        from gchain.baselines.static_gnn import StaticGNNConfig, train_and_infer_tail
        from gchain.train.streams import Stream

        n = 80
        split = 56
        msg_dim = 27
        src = torch.randint(0, 12, (n,), dtype=torch.long)
        dst = torch.randint(0, 12, (n,), dtype=torch.long)
        st = Stream(
            src=src,
            dst=dst,
            t=torch.arange(n, dtype=torch.long),
            msg=torch.randn(n, msg_dim),
            etype=torch.randint(0, 3, (n,), dtype=torch.long),
            y_ioc=torch.zeros(n, dtype=torch.long),
        )
        cfg = StaticGNNConfig(epochs=2, hidden_dim=32, num_layers=1, train_batch_size=32)
        for variant in ("graphsage", "rgcn"):
            scores, model, _ = train_and_infer_tail(
                st,
                train_end=split,
                tail_start=split,
                variant=variant,
                config=cfg,
            )
            self.assertEqual(len(scores), n - split)
            self.assertTrue(all(s >= 0.0 for s in scores))
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
