import unittest
from pathlib import Path


class TestTGNExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]

    def test_export_tgn_stream_shapes(self) -> None:
        import torch

        primary = self.repo_root / "artifacts/graphs/synthchain_sc1.full.pt"
        fallback = self.repo_root / "artifacts/graphs/synthchain_sc1_refactor_azure.full.pt"
        graph_path = primary if primary.is_file() else fallback
        if not graph_path.is_file():
            self.skipTest(f"Missing graph artifact: {primary} or {fallback}")

        obj = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False,
        )
        data = obj["data"]

        from graphcore import hetero_to_tgn_event_stream

        stream = hetero_to_tgn_event_stream(data, cat_hash_buckets=8, include_meta=False)
        self.assertEqual(stream.src.ndim, 1)
        self.assertEqual(stream.dst.ndim, 1)
        self.assertEqual(stream.t.ndim, 1)
        self.assertEqual(stream.etype.ndim, 1)
        self.assertEqual(stream.msg.ndim, 2)
        self.assertEqual(stream.src.shape[0], stream.dst.shape[0])
        self.assertEqual(stream.src.shape[0], stream.t.shape[0])
        self.assertEqual(stream.src.shape[0], stream.etype.shape[0])
        self.assertEqual(stream.src.shape[0], stream.msg.shape[0])


if __name__ == "__main__":
    unittest.main()

