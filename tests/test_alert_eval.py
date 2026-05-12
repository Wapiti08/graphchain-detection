import unittest

from graph.alert_eval import dedupe_events, precision_at_k_all, tail_alert_metrics, ScoredEvent


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


if __name__ == "__main__":
    unittest.main()
