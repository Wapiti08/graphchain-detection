import unittest

from gchain.eval.alert_eval import dedupe_events, precision_at_k_all, tail_alert_metrics, ScoredEvent
from gchain.eval.attack_reconstruct import (
    dedupe_rows_by_endpoint_pair,
    evaluate_reconstruction,
    filter_rows_to_ioc_log_sources,
    ioc_log_source_files_for_scenario,
    load_ioc_type_to_stage,
    stage_for_edge,
    stages_from_topk,
    topk_edges,
)
from graphcore.edge_meta import pick_primary_ioc_type


class TestAlertEval(unittest.TestCase):
    def test_dedupe_keeps_max_score(self) -> None:
        evs = [
            ScoredEvent("s", 1, 0, 1, 2, 0.5, 0),
            ScoredEvent("s", 1, 0, 1, 2, 1.0, 1),
        ]
        d = dedupe_events(evs)
        self.assertEqual(len(d), 1)
        self.assertAlmostEqual(d[0].score, 1.0)
        self.assertEqual(d[0].is_ioc, 1)

    def test_tail_alert_metrics_precision_at_k(self) -> None:
        rows = [
            {"scenario": "s", "t": 1, "etype": 0, "src": 1, "dst": 2, "score": 2.0, "is_ioc": 1},
            {"scenario": "s", "t": 2, "etype": 0, "src": 1, "dst": 3, "score": 1.0, "is_ioc": 0},
            {"scenario": "s", "t": 3, "etype": 0, "src": 1, "dst": 4, "score": 0.5, "is_ioc": 0},
        ]
        p = precision_at_k_all(rows, [1, 2])
        self.assertAlmostEqual(p[1], 1.0)
        self.assertAlmostEqual(p[2], 0.5)
        m = tail_alert_metrics(
            rows,
            topks=[2],
            alert_window=100,
            alert_quantile=0.5,
            alert_min_events=1,
            alert_topk_events=0,
            dedupe=True,
        )
        self.assertIn("p_at_2", m)
        self.assertGreaterEqual(float(m["num_alerts"]), 0.0)

    def test_pick_primary_ioc_type_prefers_exfil_over_attack_ip(self) -> None:
        self.assertEqual(
            pick_primary_ioc_type(["attack_ip", "data_exfiltration", "suspicious_port"]),
            "data_exfiltration",
        )

    def test_by_k_ioc_ranked_beats_global_when_ioc_sparse(self) -> None:
        rows = []
        for i in range(200):
            rows.append(
                {
                    "scenario": "s",
                    "t": i,
                    "etype": 0,
                    "src": i,
                    "dst": i + 1,
                    "score": 10.0 - i * 0.01,
                    "is_ioc": 0,
                    "source_file": "zeek_conn.csv",
                    "row_idx": i,
                    "ioc_type": "",
                }
            )
        rows.append(
            {
                "scenario": "s",
                "t": 300,
                "etype": 0,
                "src": 1,
                "dst": 2,
                "score": 0.5,
                "is_ioc": 1,
                "source_file": "azure_syslog.csv",
                "row_idx": 10,
                "ioc_type": "execution",
            }
        )
        stages_gt = {
            "observable": {"stages": ["execution", "command_and_control"]},
            "semantic": {"stages": ["execution", "command_and_control"]},
        }
        its = load_ioc_type_to_stage(__import__("pathlib").Path(__file__).resolve().parents[1])
        line_to_type = {("azure_syslog.csv", 10): "execution"}
        m = evaluate_reconstruction(
            rows=rows,
            stages_gt=stages_gt,
            ioc_type_to_stage=its,
            line_to_type=line_to_type,
            topks=[50],
            include_ioc_pool_upper_bound=True,
            alert_min_events=1,
            alert_quantile=0.0,
        )
        self.assertNotIn("execution", m["by_k"]["50"]["predicted_stages"])
        self.assertIn("execution", m["by_k_ioc_pool_upper_bound"]["50"]["predicted_stages"])
        self.assertIn("execution", m["by_alert_rule"]["predicted_stages"])

    def test_stage_for_edge_prefers_line_gt_over_csv_ioc_type(self) -> None:
        line_to_type = {("zeek_conn.csv", 2061): "data_exfiltration"}
        its = load_ioc_type_to_stage(
            __import__("pathlib").Path(__file__).resolve().parents[1]
        )
        row = {
            "ioc_type": "attack_ip",
            "source_file": "zeek_conn.csv",
            "row_idx": 2061,
            "is_ioc": 1,
        }
        self.assertEqual(
            stage_for_edge(row, ioc_type_to_stage=its, line_to_type=line_to_type),
            "exfiltration_impact",
        )

    def test_pair_dedupe_keeps_max_score_per_src_dst_etype(self) -> None:
        rows = [
            {
                "scenario": "s",
                "t": 1,
                "etype": 2,
                "src": 298,
                "dst": 227,
                "score": 1.5,
                "is_ioc": 0,
                "source_file": "eve.json",
                "row_idx": 1,
                "ioc_type": "",
            },
            {
                "scenario": "s",
                "t": 2,
                "etype": 2,
                "src": 298,
                "dst": 227,
                "score": 3.0,
                "is_ioc": 1,
                "source_file": "eve.json",
                "row_idx": 2,
                "ioc_type": "execution",
            },
            {
                "scenario": "s",
                "t": 3,
                "etype": 2,
                "src": 298,
                "dst": 211,
                "score": 2.0,
                "is_ioc": 0,
                "source_file": "eve.json",
                "row_idx": 3,
                "ioc_type": "",
            },
        ]
        kept, meta = dedupe_rows_by_endpoint_pair(rows)
        self.assertEqual(meta["n_pair_deduped_rows"], 2)
        self.assertEqual(len(topk_edges(rows, 2, endpoint_pair_dedupe=True)), 2)
        top1 = topk_edges(rows, 1, endpoint_pair_dedupe=True)[0]
        self.assertEqual(int(top1["dst"]), 227)
        self.assertAlmostEqual(float(top1["score"]), 3.0)

    def test_by_k_pair_dedupe_improves_stage_coverage(self) -> None:
        rows = []
        for i in range(200):
            rows.append(
                {
                    "scenario": "s",
                    "t": i,
                    "etype": 2,
                    "src": 298,
                    "dst": 227,
                    "score": 5.0 - i * 0.001,
                    "is_ioc": 0,
                    "source_file": "eve.json",
                    "row_idx": i,
                    "ioc_type": "",
                }
            )
        rows.append(
            {
                "scenario": "s",
                "t": 300,
                "etype": 0,
                "src": 1,
                "dst": 2,
                "score": 0.5,
                "is_ioc": 1,
                "source_file": "azure_syslog.csv",
                "row_idx": 10,
                "ioc_type": "execution",
            }
        )
        stages_gt = {
            "observable": {"stages": ["execution", "command_and_control"]},
            "semantic": {"stages": ["execution", "command_and_control"]},
        }
        its = load_ioc_type_to_stage(__import__("pathlib").Path(__file__).resolve().parents[1])
        line_to_type = {("azure_syslog.csv", 10): "execution"}
        plain = stages_from_topk(
            rows, k=50, ioc_type_to_stage=its, line_to_type=line_to_type, endpoint_pair_dedupe=False
        )
        paired = stages_from_topk(
            rows, k=50, ioc_type_to_stage=its, line_to_type=line_to_type, endpoint_pair_dedupe=True
        )
        self.assertNotIn("execution", plain)
        self.assertIn("execution", paired)

    def test_ioc_log_source_filter(self) -> None:
        allowed = ioc_log_source_files_for_scenario("sc3")
        self.assertIn("azure_syslog.csv", allowed)
        rows = [
            {"source_file": "azure_syslog.csv", "score": "1.0", "is_ioc": "1"},
            {"source_file": "noise.log", "score": "9.0", "is_ioc": "0"},
        ]
        kept, meta = filter_rows_to_ioc_log_sources(rows, allowed)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["source_file"], "azure_syslog.csv")
        self.assertEqual(meta["n_excluded_rows"], 1)

    def test_by_k_pred_stage_uses_all_topk_edges(self) -> None:
        rows = [
            {
                "scenario": "s",
                "t": 1,
                "etype": 0,
                "src": 1,
                "dst": 2,
                "score": 10.0,
                "is_ioc": 0,
                "pred_stage": "command_and_control",
                "pred_stage_prob": 0.95,
            },
            {
                "scenario": "s",
                "t": 2,
                "etype": 0,
                "src": 1,
                "dst": 3,
                "score": 9.0,
                "is_ioc": 0,
                "pred_stage": "execution",
                "pred_stage_prob": 0.9,
            },
        ]
        stages_gt = {
            "observable": {"stages": ["execution", "command_and_control"]},
            "semantic": {"stages": ["execution", "command_and_control"]},
        }
        its = load_ioc_type_to_stage(__import__("pathlib").Path(__file__).resolve().parents[1])
        m = evaluate_reconstruction(
            rows=rows,
            stages_gt=stages_gt,
            ioc_type_to_stage=its,
            line_to_type={},
            topks=[2],
            run_alert_reconstruction=False,
        )
        pred = set(m["by_k_pred_stage"]["2"]["predicted_stages"])
        self.assertEqual(pred, {"execution", "command_and_control"})
        self.assertNotIn("execution", m["by_k"]["2"]["predicted_stages"])

    def test_by_alert_pred_stage_clusters_high_score_events(self) -> None:
        rows = []
        for i in range(4):
            rows.append(
                {
                    "scenario": "s",
                    "t": 100 + i,
                    "etype": 0,
                    "src": 1,
                    "dst": 2 + i,
                    "score": 5.0,
                    "is_ioc": 0,
                    "pred_stage": "command_and_control",
                    "pred_stage_prob": 0.9,
                }
            )
        rows.append(
            {
                "scenario": "s",
                "t": 200,
                "etype": 0,
                "src": 9,
                "dst": 10,
                "score": 0.1,
                "is_ioc": 0,
                "pred_stage": "execution",
                "pred_stage_prob": 0.9,
            }
        )
        stages_gt = {
            "observable": {"stages": ["command_and_control"]},
            "semantic": {"stages": ["command_and_control", "execution"]},
        }
        its = load_ioc_type_to_stage(__import__("pathlib").Path(__file__).resolve().parents[1])
        m = evaluate_reconstruction(
            rows=rows,
            stages_gt=stages_gt,
            ioc_type_to_stage=its,
            line_to_type={},
            topks=[10],
            alert_min_events=3,
            alert_quantile=0.5,
        )
        self.assertIn("command_and_control", m["by_alert_pred_stage"]["predicted_stages"])
        self.assertNotIn("execution", m["by_alert_pred_stage"]["predicted_stages"])


if __name__ == "__main__":
    unittest.main()
