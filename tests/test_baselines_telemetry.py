"""Telemetry filters for Scheme A single-source baselines."""
from __future__ import annotations

import unittest

from gchain.baselines.telemetry import (
    Telemetry,
    classify_source_file,
    edge_mask_for_telemetry,
    scenario_supports_telemetry,
)


class TestBaselinesTelemetry(unittest.TestCase):
    def test_classify_source_file(self) -> None:
        self.assertEqual(classify_source_file("windows/azure_events.csv"), Telemetry.AUDIT)
        self.assertEqual(classify_source_file("zeek_http.csv"), Telemetry.ZEEK)
        self.assertEqual(classify_source_file("eve.json"), Telemetry.EVE)

    def test_edge_mask_audit(self) -> None:
        sf = ("azure_events.csv", "zeek_conn.csv", "eve.json")
        mask = edge_mask_for_telemetry(sf, "audit")
        self.assertEqual(mask, [True, False, False])

    def test_scenario_eve_na(self) -> None:
        self.assertFalse(scenario_supports_telemetry("sc2", "eve"))
        self.assertTrue(scenario_supports_telemetry("sc3", "eve"))


if __name__ == "__main__":
    unittest.main()
