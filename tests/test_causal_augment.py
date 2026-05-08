import unittest
from pathlib import Path

from graph.augment import augment_events_with_causal
from parsers.qut.processed import parse_syscall_row


class TestCausalAugment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def test_qut_events_can_get_causal_edges_without_ts(self) -> None:
        # QUT processed has no explicit ts; uses order.
        import pandas as pd

        p = (
            self.repo_root
            / "data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets/"
            "QUT-DV25_SysCall_Traces/QUT-DV25_SysCall_Traces.csv"
        )
        df = pd.read_csv(p).head(1)
        events = parse_syscall_row(df.iloc[0])
        aug = augment_events_with_causal(events, mode="level0", window=100.0)

        # should add at least one causal edge in most cases
        n_added = len(aug) - len(events)
        self.assertGreaterEqual(n_added, 1)

        # no negative delta_t
        for e in aug:
            if e.edge_type.value == "CAUSE":
                self.assertGreaterEqual(float(e.edge_attrs.get("delta_t", 0)), 0.0)


if __name__ == "__main__":
    unittest.main()

