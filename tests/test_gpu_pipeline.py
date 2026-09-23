from __future__ import annotations

import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from nailong.device import checked_device
from nailong.stages import build_dataset


class GpuPipelineTest(unittest.TestCase):
    def test_cuda_request_fails_without_cuda_instead_of_falling_back(self) -> None:
        fake_torch = types.SimpleNamespace(
            __version__="test+cpu",
            cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(SystemExit, "无法使用该 GPU"):
                checked_device("cuda:0")
        self.assertEqual(checked_device("cpu"), "cpu")

    def test_plan_routes_device_to_all_model_stages(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(build_dataset.main(["--plan", "--device", "cuda:1"]), 0)
        plan = output.getvalue()
        for stage in ("separate", "transcribe", "embed-redimnet", "embed-eres2net"):
            self.assertIn(f"{stage} --device cuda:1", plan)
        self.assertIn("segment", plan)

    def test_plan_uses_selected_candidate_ranking(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(build_dataset.main([
                "--plan", "--batch", "3", "--ranking", "candidate_scores.csv",
            ]), 0)
        self.assertIn("--limit 3 --ranking candidate_scores.csv", output.getvalue())


if __name__ == "__main__":
    unittest.main()
