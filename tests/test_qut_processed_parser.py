import unittest
from pathlib import Path

import pandas as pd

from config.ontology import EdgeType, NodeType
from parsers.events import Event
from parsers.qut.processed import (
    parse_filetop_row,
    parse_opensnoop_row,
    parse_syscall_row,
)
from parsers.qut.join import apply_segmented_order, dedup_load

class TestQUTProcessedParsers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _assert_event_basic(self, ev: Event) -> None:
        self.assertIsInstance(ev, Event)
        self.assertIsInstance(ev.edge_type, EdgeType)
        self.assertIsInstance(ev.src.type, NodeType)
        self.assertIsInstance(ev.dst.type, NodeType)
        self.assertIsInstance(ev.src.key, str)
        self.assertIsInstance(ev.dst.key, str)
        self.assertIsInstance(ev.edge_attrs, dict)

    def test_syscall_row_parses_into_load_and_invokes(self) -> None:
        p = (
            self.repo_root
            / "data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets/"
            "QUT-DV25_SysCall_Traces/QUT-DV25_SysCall_Traces.csv"
        )
        df = pd.read_csv(p)
        events = parse_syscall_row(df.iloc[0])
        self.assertGreaterEqual(len(events), 2)

        self._assert_event_basic(events[0])
        self.assertEqual(events[0].edge_type, EdgeType.LOAD)
        self.assertEqual(events[0].src.type, NodeType.PKG)
        self.assertEqual(events[0].dst.type, NodeType.PROC)

        invoke_events = [e for e in events if e.edge_type == EdgeType.INVOKE]
        self.assertGreaterEqual(len(invoke_events), 1)
        for e in invoke_events[:5]:
            self._assert_event_basic(e)
            self.assertEqual(e.src.type, NodeType.PROC)
            self.assertEqual(e.dst.type, NodeType.SYSCALL)

    def test_segmented_order_is_deterministic(self) -> None:
        p = (
            self.repo_root
            / "data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets/"
            "QUT-DV25_SysCall_Traces/QUT-DV25_SysCall_Traces.csv"
        )
        df = pd.read_csv(p)
        ev = parse_syscall_row(df.iloc[0])
        ev = dedup_load(ev)
        a = apply_segmented_order(ev)
        b = apply_segmented_order(ev)
        self.assertEqual([x.order for x in a], [x.order for x in b])

    def test_opensnoop_row_parses_into_bucket_writes(self) -> None:
        p = (
            self.repo_root
            / "data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets/"
            "QUT-DV25_Opensnoop_Traces/QUT-DV25_Opensnoop_Traces.csv"
        )
        df = pd.read_csv(p)
        events = parse_opensnoop_row(df.iloc[0])
        self.assertGreaterEqual(len(events), 1)

        self.assertEqual(events[0].edge_type, EdgeType.LOAD)
        write_events = [e for e in events if e.edge_type == EdgeType.WRITE]
        self.assertGreaterEqual(len(write_events), 1)
        for e in write_events[:5]:
            self._assert_event_basic(e)
            self.assertEqual(e.src.type, NodeType.PROC)
            self.assertEqual(e.dst.type, NodeType.FILE)

    def test_filetop_row_parses_reads_writes_and_execs(self) -> None:
        p = (
            self.repo_root
            / "data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets/"
            "QUT-DV25_Filetop_Traces/QUT-DV25_Filetop_Traces.csv"
        )
        df = pd.read_csv(p)
        events = parse_filetop_row(df.iloc[0])
        self.assertGreaterEqual(len(events), 2)

        self.assertEqual(events[0].edge_type, EdgeType.LOAD)
        for e in events[:10]:
            self._assert_event_basic(e)

        # Should include at least one of READ/WRITE or EXEC depending on row values
        edge_types = {e.edge_type for e in events}
        self.assertTrue(
            (EdgeType.READ in edge_types)
            or (EdgeType.WRITE in edge_types)
            or (EdgeType.EXEC in edge_types)
        )


if __name__ == "__main__":
    unittest.main()

