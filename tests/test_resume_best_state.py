"""A device takeover must retain the best epoch when resuming from last.pt."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train


class BestStateResumeTests(unittest.TestCase):
    def test_last_state_does_not_replace_the_saved_best(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(3.0)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            torch.save({"epoch": 46, "model_state_dict": {"weight": torch.tensor([[1.0]])}},
                       run / "best.pt")
            state = train.best_state_for_resume(model, run)
            self.assertEqual(state["weight"].item(), 1.0)
            self.assertEqual(model.weight.item(), 3.0)

    def test_missing_best_falls_back_to_an_independent_copy(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with tempfile.TemporaryDirectory() as directory:
            state = train.best_state_for_resume(model, Path(directory))
            self.assertTrue(torch.equal(state["weight"], model.weight))
            with torch.no_grad():
                model.weight.add_(1)
            self.assertFalse(torch.equal(state["weight"], model.weight))


if __name__ == "__main__":
    unittest.main()
