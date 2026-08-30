import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl

import run
from match.backend_routing import resolve_effective_backend
from match.submission import _solution_with_effective_backend


class BackendRoutingTests(unittest.TestCase):
    def test_ort_tensorrt_provider_prepares_tensorrt_runtime(self) -> None:
        self.assertTrue(
            run._requires_tensorrt_runtime(
                {
                    "backend": "onnxruntime",
                    "onnxruntime": {"provider": "tensorrt"},
                }
            )
        )
        self.assertFalse(
            run._requires_tensorrt_runtime(
                {
                    "backend": "onnxruntime",
                    "onnxruntime": {"provider": "cuda"},
                }
            )
        )

    def test_evaluator_invocation_falls_back_to_packaged_solution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packaged_solution = root / "solution.json"
            packaged_solution.write_text("{}", encoding="utf-8")
            args = SimpleNamespace(solution=None, config=str(root / "missing.yaml"))
            with patch.object(run, "__file__", str(root / "run.py")):
                resolved = run._predict_solution_path(args)
        self.assertEqual(resolved, packaged_solution)

    def test_missing_packaged_solution_fails_before_runtime_setup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(solution=None, config=str(root / "missing.yaml"))
            with (
                patch.object(run, "__file__", str(root / "run.py")),
                self.assertRaisesRegex(FileNotFoundError, "packaged manifest"),
            ):
                run._predict_solution_path(args)

    def test_resolves_threshold_boundary_once(self) -> None:
        solution = {"backend": "adaptive", "adaptive_pair_threshold": 10_000}
        self.assertEqual(
            resolve_effective_backend(solution, 10_000),
            "onnxruntime",
        )
        self.assertEqual(
            resolve_effective_backend(solution, 10_001),
            "tensorrt",
        )

    def test_rejects_preflight_and_predictor_backend_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime backend mismatch"):
            _solution_with_effective_backend(
                {"backend": "adaptive", "adaptive_pair_threshold": 10_000},
                1_000,
                "tensorrt",
            )

    def _run_predict(self, pair_count: int) -> list[str]:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solution = root / "solution.json"
            matches = root / "matches.parquet"
            solution.write_text(
                json.dumps(
                    {
                        "backend": "adaptive",
                        "adaptive_pair_threshold": 10_000,
                    }
                ),
                encoding="utf-8",
            )
            pl.DataFrame(
                {"id1": range(pair_count), "id2": range(pair_count)}
            ).write_parquet(matches)
            args = SimpleNamespace(
                solution=str(solution),
                config="missing.yaml",
                items_path="items.parquet",
                matches_path=str(matches),
                output_path="submit.csv",
            )

            def ensure_ort(*args, backend_override=None, **kwargs):
                del args, kwargs
                if backend_override == "onnxruntime":
                    events.append("runtime:onnxruntime")

            def ensure_trt(*args, backend_override=None, **kwargs):
                del args, kwargs
                if backend_override == "tensorrt":
                    events.append("runtime:tensorrt")

            def preprocessing(*args, **kwargs):
                del args, kwargs
                events.append("preprocessing")

            def create_submission(*args, backend_override=None, **kwargs):
                del args, kwargs
                events.append(f"predictor:{backend_override}")
                return SimpleNamespace(height=pair_count)

            with (
                patch.object(run, "ensure_polars_available"),
                patch.object(run, "ensure_onnxruntime_available", ensure_ort),
                patch.object(run, "ensure_tensorrt_available", ensure_trt),
                patch.object(
                    run,
                    "ensure_preprocessing_runtime_available",
                    preprocessing,
                ),
                patch.object(run, "ensure_catboost_available"),
                patch("match.submission.create_submission", create_submission),
            ):
                run.run_predict(args)
        return events

    def test_small_input_validates_ort_before_preprocessing(self) -> None:
        self.assertEqual(
            self._run_predict(1_000),
            ["runtime:onnxruntime", "preprocessing", "predictor:onnxruntime"],
        )

    def test_large_input_validates_tensorrt_before_preprocessing(self) -> None:
        self.assertEqual(
            self._run_predict(10_001),
            ["runtime:tensorrt", "preprocessing", "predictor:tensorrt"],
        )


if __name__ == "__main__":
    unittest.main()
