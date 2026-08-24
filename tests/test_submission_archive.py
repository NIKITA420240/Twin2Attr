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
            features=replace(
                config.features,
                normalization=replace(
                    config.features.normalization,
                    enabled=False,
                ),
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
        (wheels / "pymorphy3-2.0.6-py3-none-any.whl").write_bytes(b"pymorphy3")
        (
            wheels
            / "pymorphy3_dicts_ru-2.4.417150.4580142-py2.py3-none-any.whl"
        ).write_bytes(b"dicts")
        (wheels / "dawg2_python-0.9.0-py3-none-any.whl").write_bytes(b"dawg")
        (wheels / "setuptools-84.0.0-py3-none-any.whl").write_bytes(b"setuptools")
        (wheels / "joblib-1.5.3-py3-none-any.whl").write_bytes(b"joblib")
        (wheels / "loguru-0.7.3-py3-none-any.whl").write_bytes(b"loguru")
        (wheels / "Pint-0.25.3-py3-none-any.whl").write_bytes(b"pint")
        (wheels / "flexcache-0.3-py3-none-any.whl").write_bytes(b"flexcache")
        (wheels / "flexparser-0.4-py3-none-any.whl").write_bytes(b"flexparser")
        (wheels / "platformdirs-4.11.3-py3-none-any.whl").write_bytes(
            b"platformdirs"
        )
        (wheels / "typing_extensions-4.16.0-py3-none-any.whl").write_bytes(
            b"typing-extensions"
        )
        (wheels / "colorama-0.4.6-py2.py3-none-any.whl").write_bytes(b"colorama")
        (wheels / "win32_setctime-1.2.0-py3-none-any.whl").write_bytes(
            b"win32-setctime"
        )

    @staticmethod
    def _trained_transformer(root: Path) -> Path:
        transformer = root / "models" / "transformer"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["BertForSequenceClassification"],
                    "id2label": {"0": "different", "1": "match"},
                    "match_max_length": 32,
                    "match_use_field_tokens": False,
                }
            ),
            encoding="utf-8",
        )
        (transformer / "model.safetensors").write_bytes(b"weights")
        return transformer

    def test_packages_only_selected_model_and_generates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._project(root)
            transformer = self._trained_transformer(root)
            checkpoint = transformer / "checkpoint-10"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"duplicate")
            maxpooling = root / "models" / "maxpooling.joblib"
            synonyms = root / "data" / "synonyms.parquet"
            synonyms.parent.mkdir(parents=True)
            synonyms.write_bytes(b"synonyms")
            unique = root / "data" / "unique.parquet"
            unique.write_bytes(b"unique")
            output = root / "dist" / "submission.zip"
            config = replace(
                self.config,
                inference=replace(
                    self.config.inference,
                    transformer_dir=transformer,
                    maxpooling_path=maxpooling,
                ),
                features=replace(
                    self.config.features,
                    normalization=replace(
                        self.config.features.normalization,
                        enabled=True,
                        synonyms_path=synonyms,
                        unique_attributes_path=unique,
                    ),
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
        self.assertIn("vendor_wheels/pymorphy3-2.0.6-py3-none-any.whl", names)
        self.assertIn(
            "vendor_wheels/pymorphy3_dicts_ru-2.4.417150.4580142-py2.py3-none-any.whl",
            names,
        )
        self.assertIn("vendor_wheels/dawg2_python-0.9.0-py3-none-any.whl", names)
        self.assertIn("vendor_wheels/setuptools-84.0.0-py3-none-any.whl", names)
        self.assertIn("vendor_wheels/joblib-1.5.3-py3-none-any.whl", names)
        self.assertIn("vendor_wheels/loguru-0.7.3-py3-none-any.whl", names)
        self.assertIn("vendor_wheels/Pint-0.25.3-py3-none-any.whl", names)
        self.assertIn("vendor_wheels/flexcache-0.3-py3-none-any.whl", names)
        self.assertIn("vendor_wheels/flexparser-0.4-py3-none-any.whl", names)
        self.assertIn("vendor_wheels/platformdirs-4.11.3-py3-none-any.whl", names)
        self.assertIn(
            "vendor_wheels/typing_extensions-4.16.0-py3-none-any.whl",
            names,
        )
        self.assertIn("vendor_wheels/colorama-0.4.6-py2.py3-none-any.whl", names)
        self.assertIn(
            "vendor_wheels/win32_setctime-1.2.0-py3-none-any.whl",
            names,
        )
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
            transformer = self._trained_transformer(root)
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
                features=replace(
                    self.config.features,
                    normalization=replace(
                        self.config.features.normalization,
                        enabled=True,
                        synonyms_path=synonyms,
                        unique_attributes_path=unique,
                    ),
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
        self.assertNotIn("normalization", solution)
        self.assertEqual(
            solution["features"]["execution_order"],
            ["normalization", "ner", "physical"],
        )
        self.assertTrue(solution["features"]["normalization"]["enabled"])
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

    def test_rejects_raw_pretrained_transformer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._project(root)
            transformer = root / "models" / "raw-transformer"
            transformer.mkdir(parents=True)
            (transformer / "config.json").write_text(
                json.dumps({"model_type": "distilbert"}),
                encoding="utf-8",
            )
            (transformer / "pytorch_model.bin").write_bytes(b"pretrained")
            config = replace(
                self.config,
                inference=replace(
                    self.config.inference,
                    transformer_dir=transformer,
                ),
                submission=replace(
                    self.config.submission,
                    output_path=root / "submission.zip",
                ),
            )

            with self.assertRaisesRegex(ValueError, "trained Twin2Attr classifier"):
                build_submission_archive(config, project_root=root)

    def test_cascade_packages_both_models_and_catboost_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._project(root)
            transformer = self._trained_transformer(root)
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
