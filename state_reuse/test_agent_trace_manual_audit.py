import argparse
import csv
import json
import os
import tempfile
import unittest

import agent_trace_manual_audit as audit


class ManualAuditTest(unittest.TestCase):
    def test_proportional_allocation(self):
        allocation = audit.proportional_allocation({"a": 70, "b": 20, "c": 10}, 20)
        self.assertEqual(allocation, {"a": 14, "b": 4, "c": 2})

    def test_metric_ambiguity_bounds(self):
        rows = [
            self.metric_row("safe", 100, 50),
            self.metric_row("history_dependent", 100, 80),
            self.metric_row("ambiguous", 200, 100),
            self.metric_row("bookkeeping", 500, 500),
        ]
        metrics = audit.metric_bounds(rows)
        self.assertAlmostEqual(metrics["f_safe_lower"], 1 / 3)
        self.assertAlmostEqual(metrics["f_safe_upper"], 2 / 3)
        self.assertAlmostEqual(metrics["oracle_saving_lower"], 50 / 400)
        self.assertAlmostEqual(metrics["oracle_saving_upper"], 150 / 400)

    def test_duplicate_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "labels.csv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=("item_id", "manual_label"))
                writer.writeheader()
                writer.writerow({"item_id": "same", "manual_label": "safe"})
                writer.writerow({"item_id": "same", "manual_label": "ambiguous"})
            with self.assertRaisesRegex(ValueError, "duplicate labeled item_id"):
                audit.read_packet_labels(path)

    def test_prepare_and_analyze(self):
        with tempfile.TemporaryDirectory() as tmp:
            turns_path = os.path.join(tmp, "turns.jsonl")
            turns = []
            for i in range(12):
                turns.append(
                    {
                        "key": f"{i // 3}:{i}",
                        "row_idx": i // 3,
                        "instance_id": f"instance-{i // 3}",
                        "turn_index": i + 1,
                        "command_family": "edit" if i < 6 else "inspect",
                        "target": bool(i % 2),
                        "context_tokens_est": 100 + i,
                        "old_history_tokens_est": 50 + i,
                        "task_text": "task",
                        "old_history": "history",
                        "latest_observation": "observation",
                        "ai_text": "generation",
                    }
                )
            audit.write_jsonl(turns_path, turns)
            prep_out = os.path.join(tmp, "prepared")
            audit.prepare(
                argparse.Namespace(turns=turns_path, sample_size=8, overlap_size=4, seed=0, out=prep_out)
            )
            manifest = audit.read_jsonl(os.path.join(prep_out, "sample_manifest.jsonl"))
            self.assertEqual(len(manifest), 8)
            self.assertEqual(sum(row["is_overlap"] for row in manifest), 4)
            self.assertEqual(sum(row["in_labeler_a"] for row in manifest), 6)
            self.assertEqual(sum(row["in_labeler_b"] for row in manifest), 6)

            labels_a = os.path.join(tmp, "labels_a.csv")
            labels_b = os.path.join(tmp, "labels_b.csv")
            self.write_labels(labels_a, manifest, "in_labeler_a")
            self.write_labels(labels_b, manifest, "in_labeler_b")
            analysis_out = os.path.join(tmp, "analysis")
            audit.analyze(
                argparse.Namespace(
                    manifest=os.path.join(prep_out, "sample_manifest.jsonl"),
                    labels_a=labels_a,
                    labels_b=labels_b,
                    adjudication=None,
                    bootstrap_trials=100,
                    seed=0,
                    out=analysis_out,
                )
            )
            with open(os.path.join(analysis_out, "summary.json"), "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["resolved_items"], 8)
            self.assertEqual(summary["unresolved_items"], 0)
            self.assertEqual(summary["raw_agreement"], 1.0)
            self.assertEqual(summary["counts"], {"safe": 8})

    @staticmethod
    def metric_row(label, context, old_history):
        return {
            "final_label": label,
            "sampling_weight": 1.0,
            "context_tokens_est": context,
            "old_history_tokens_est": old_history,
        }

    @staticmethod
    def write_labels(path, manifest, membership):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("item_id", "manual_label"))
            writer.writeheader()
            for row in manifest:
                if row[membership]:
                    writer.writerow({"item_id": row["item_id"], "manual_label": "safe"})


if __name__ == "__main__":
    unittest.main()
