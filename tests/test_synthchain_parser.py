import unittest
from pathlib import Path

from config.ontology import EdgeType, NodeType
from parsers.events import Event
from parsers.synthchain import load_synthchain_events


class TestSynthChainParsers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _assert_event_basic(self, ev: Event) -> None:
        self.assertIsInstance(ev, Event)
        self.assertIsInstance(ev.edge_type, EdgeType)
        self.assertIsInstance(ev.src.type, NodeType)
        self.assertIsInstance(ev.dst.type, NodeType)
        self.assertIsInstance(ev.edge_attrs, dict)

    def test_sc1_ioc_logs_parse(self) -> None:
        # sc1 uses Azure CSVs; keep limits small for fast tests
        events = load_synthchain_events(
            "sc1", project_root=self.repo_root, only_ioc_logs=True, limit_per_file=50
        )
        print(events)
        self.assertGreater(len(events), 0)
        for e in events[:20]:
            self._assert_event_basic(e)
        # Expect at least one CONNECT from azure_conn
        self.assertTrue(any(e.edge_type == EdgeType.CONNECT for e in events))

    def test_sc3_ioc_logs_parse(self) -> None:
        # sc3 includes zeek CSVs and eve.json; use tiny limit per file
        events = load_synthchain_events(
            "sc3", project_root=self.repo_root, only_ioc_logs=True, limit_per_file=30
        )
        print(events)

        self.assertGreater(len(events), 0)
        for e in events[:20]:
            self._assert_event_basic(e)
        # Expect at least one of CONNECT/DNS_QUERY/RESOLVE edges
        edge_types = {e.edge_type for e in events}
        self.assertTrue(
            (EdgeType.CONNECT in edge_types)
            or (EdgeType.DNS_QUERY in edge_types)
            or (EdgeType.RESOLVE in edge_types)
        )


if __name__ == "__main__":
    unittest.main()

