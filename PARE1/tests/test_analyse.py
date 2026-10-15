import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyse_predictions import read_predictions, score, compare, percentile


class AuditTests(unittest.TestCase):
    def save_csv(self, records):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "pred.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "source_id", "sample_id", "split", "y_true", "resnet_pred", "deit_pred"
            ])
            writer.writeheader()
            writer.writerows(records)
        self.addCleanup(directory.cleanup)
        return path

    def sample(self, src, sample, split="test", truth="A", a="A", b="A"):
        return dict(source_id=src, sample_id=sample, split=split,
                    y_true=truth, resnet_pred=a, deit_pred=b)

    def test_accuracy(self):
        rows = [self.sample("x", "1"), self.sample("y", "2", truth="B", a="A", b="B")]
        self.assertEqual(score(rows, "resnet_pred")["accuracy"], 0.5)
        self.assertEqual(score(rows, "deit_pred")["balanced_accuracy"], 1.0)

    def test_partition_leak_detected(self):
        path = self.save_csv([self.sample("x", "1", "train"),
                              self.sample("x", "2", "test")])
        with self.assertRaisesRegex(ValueError, "Fuite"):
            read_predictions(path)

    def test_duplicate_detected(self):
        path = self.save_csv([self.sample("x", "1"), self.sample("y", "1")])
        with self.assertRaisesRegex(ValueError, "dupliqué"):
            read_predictions(path)

    def test_deterministic_bootstrap(self):
        rows = [self.sample("x", "1", a="B"),
                self.sample("y", "2", truth="B", a="B", b="B")]
        self.assertEqual(compare(rows, 100, 7), compare(rows, 100, 7))

    def test_single_source_no_interval(self):
        rows = [self.sample("x", "1")]
        self.assertIsNone(compare(rows, 100)["cluster_bootstrap_95pct_ci"])

    def test_percentile(self):
        self.assertEqual(percentile([0, 10], 0.25), 2.5)


if __name__ == "__main__":
    unittest.main()
