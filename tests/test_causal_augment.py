import unittest

from config.ontology import EdgeType, NodeType
from graphcore.augment import augment_events_with_causal
from parsers.events import EntityRef, Event


class TestCausalAugment(unittest.TestCase):
    def test_events_without_ts_can_get_causal_edges(self) -> None:
        proc = EntityRef(NodeType.PROC, "p1")
        file_a = EntityRef(NodeType.FILE, "f1")
        file_b = EntityRef(NodeType.FILE, "f2")
        events = [
            Event(EdgeType.READ, proc, file_a, ts=None, order=0),
            Event(EdgeType.WRITE, proc, file_b, ts=None, order=1),
        ]
        aug = augment_events_with_causal(events, mode="level0", window=100.0)

        n_added = len(aug) - len(events)
        self.assertGreaterEqual(n_added, 1)

        for e in aug:
            if e.edge_type.value == "CAUSE":
                self.assertGreaterEqual(float(e.edge_attrs.get("delta_t", 0)), 0.0)


if __name__ == "__main__":
    unittest.main()
