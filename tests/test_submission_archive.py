import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

from build_submission.archive import build_submission_archive
from match.config import load_app_config_file
from match.paths import PROJECT_ROOT


class SubmissionArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml"
        )
        self.config = replace(
            config,
            normalization=replace(config.normalization, enabled=False),
            features=replace(
                config.features,
                ner=replace(config.features.ner, enabled=False),
            ),
        )

    @staticmethod
    def _project(root: Path) -> None:
        (root / "src" / "match").mkdir(parents=True)
        (root / "src" / "match" / "__init__.py").write_text("", encoding="utf-8")
        (root / "src" / "match" / "module.py").write_text("VALUE = 1\n")
        (root / "src" / "match" / "__pycache__").mkdir()
        (root / "src" / "match" / "__pycache__" / "module.pyc").write_bytes(
            b"cache"
        )
        (root / "run.py").write_text("print('run')\n", encoding="utf-8")
        (root / "metadata.json").write_text("{}\n", encoding="utf-8")
        wheels = root / "build_submission" / "vendor_wheels"
        wheels.mkdir(parents=True)
        (wheels / "polars-1.43.2-py3-none-any.whl").write_bytes(b"polars")
        (
            wheels
            / "polars_runtime_32-1.43.2-cp310-abi3-manylinux_x86_64.whl"
        ).write_bytes(b"runtime")
        (wheels / "catboost-1.2.10-cp312-cp312-manylinux_x86_64.whl").write_bytes(
            b"catboost"
        )

    def test_packages_only_selected_model_and_generates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._project(root)
            transformer = root / "models" / "transformer"
            transformer.mkdir(parents=True)
            (transformer / "config.json").write_text("{}", encoding="utf-8")
            (transformer / "model.safetensors").write_bytes(b"weights")
            checkpoint = transformer / "checkpoint-10"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"duplicate")
            maxpooling = root / "models" / "maxpooling.joblib"
            output = root / "dist" / "submission.zip"
            config = replace(
                self.config,
                inference=replace(
                    self.config.inference,
                    transformer_dir=transformer,
                    maxpooling_path=maxpooling,
                ),
                submission=replace(
                    self.config.submission,
                    output_path=output,
                ),
            )

            result = build_submission_archive(config, project_root=root)

            with ZipFile(result.path) as archive:
                names = set(archive.namelist())
                solution = json.loads(archive.read("solution.json"))

        self.assertEqual(result.predictor, "transformer")
        self.assertEqual(solution["predictor"], "transformer")
        self.assertEqual(solution["model_directory"], "models/transformer")
        self.assertIn("models/transformer/model.safetensors", names)
        self.assertIn("vendor_wheels/polars-1.43.2-py3-none-any.whl", names)
        self.assertNotIn(
            "vendor_wheels/catboost-1.2.10-cp312-cp312-manylinux_x86_64.whl",
            names,
        )
        self.assertNotIn("models/maxpooling.joblib", names)
        self.assertNotIn(
            "models/transformer/checkpoint-10/model.safetensors",
            names,
        )
        self.assertFalse(any("__pycache__" in name for name in names))

    def test_packages_enabled_preprocessing_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._project(root)
            transformer = root / "models" / "transformer"
            transformer.mkdir(parents=True)
            (transformer / "config.json").write_text("{}", encoding="utf-8")
            ner_dir = root / "models" / "ner"
            ner_dir.mkdir(parents=True)
            (ner_dir / "model.pt").write_bytes(b"ner")
            centers = root / "models" / "cluster_centers.pt"
            centers.write_bytes(b"centers")
            synonyms = root / "data" / "synonyms.parquet"
            synonyms.parent.mkdir(parents=True)
            synonyms.write_bytes(b"synonyms")
            unique = root / "data" / "unique.parquet"
            unique.write_bytes(b"unique")
            config = replace(
                self.config,
                inference=replace(
                    self.config.inference,
                    transformer_dir=transformer,
                ),
                normalization=replace(
                    self.config.normalization,
                    enabled=True,
                    synonyms_path=synonyms,
                    unique_attributes_path=unique,
                ),
                features=replace(
                    self.config.features,
                    ner=replace(
                        self.config.features.ner,
                        enabled=True,
                        model_dir=ner_dir,
                        cluster_centers_path=centers,
                    ),
                    physical=replace(
                        self.config.features.physical,
                        enabled=True,
                    ),
                ),
                submission=replace(
                    self.config.submission,
                    output_path=root / "submission.zip",
                ),
            )

            result = build_submission_archive(config, project_root=root)

            with ZipFile(result.path) as archive:
                names = set(archive.namelist())
                solution = json.loads(archive.read("solution.json"))

        self.assertIn("data/synonyms.parquet", names)
        self.assertIn("data/unique.parquet", names)
        self.assertIn("models/ner/model.pt", names)
        self.assertIn("models/cluster_centers.pt", names)
        self.assertTrue(solution["normalization"]["enabled"])
        self.assertTrue(solution["features"]["ner"]["enabled"])
        self.assertTrue(solution["features"]["physical"]["enabled"])

    def test_maxpooling_does_not_require_transformer_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._project(root)
            maxpooling = root / "models" / "maxpooling.joblib"
            maxpooling.parent.mkdir(parents=True)
            maxpooling.write_bytes(b"maxpooling")
            config = replace(
                self.config,
                inference=replace(
                    self.config.inference,
                    model="maxpooling",
                    transformer_dir=root / "models" / "missing-transformer",
                    maxpooling_path=maxpooling,
                ),
                submission=replace(
                    self.config.submission,
                    output_path=root / "submission.zip",
                ),
            )

            result = build_submission_archive(config, project_root=root)

            with ZipFile(result.path) as archive:
                names = set(archive.namelist())
                solution = json.loads(archive.read("solution.json"))

        self.assertEqual(solution["predictor"], "maxpooling")
        self.assertEqual(solution["maxpooling_path"], "models/maxpooling.joblib")
        self.assertIn("models/maxpooling.joblib", names)
        self.assertFalse(any("missing-transformer" in name for name in names))

    def test_fails_when_selected_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._project(root)
            config = replace(
                self.config,
                inference=replace(
                    self.config.inference,
                    transformer_dir=root / "models" / "missing",
                ),
                submission=replace(
                    self.config.submission,
                    output_path=root / "submission.zip",
                ),
            )

            with self.assertRaisesRegex(FileNotFoundError, "Train the selected"):
                build_submission_archive(config, project_root=root)

    def test_cascade_packages_both_models_and_catboost_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._project(root)
            transformer = root / "models" / "transformer"
            transformer.mkdir(parents=True)
            (transformer / "model.safetensors").write_bytes(b"transformer")
            boosting = root / "models" / "boosting"
            boosting.mkdir(parents=True)
            (boosting / "model.cbm").write_bytes(b"boosting")
            (boosting / "manifest.json").write_text("{}", encoding="utf-8")
            config = replace(
                self.config,
                inference=replace(
                    self.config.inference,
                    model="cascade",
                    transformer_dir=transformer,
                    boosting_dir=boosting,
                ),
                submission=replace(
                    self.config.submission,
                    output_path=root / "submission.zip",
                ),
            )

            result = build_submission_archive(config, project_root=root)

            with ZipFile(result.path) as archive:
                names = set(archive.namelist())
                solution = json.loads(archive.read("solution.json"))

        self.assertEqual(solution["predictor"], "cascade")
        self.assertEqual(solution["fast_model"], "boosting")
        self.assertEqual(solution["main_model"], "transformer")
        self.assertIn("models/transformer/model.safetensors", names)
        self.assertIn("models/boosting/model.cbm", names)
        self.assertIn(
            "vendor_wheels/catboost-1.2.10-cp312-cp312-manylinux_x86_64.whl",
            names,
        )


if __name__ == "__main__":
    unittest.main()
