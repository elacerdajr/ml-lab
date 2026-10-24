"""Runs and the experiment record."""

import json

import pandas as pd
import pytest

from ml_elements.run import ProfileMismatch, Run, record, report_figures


def test_default_profile_writes_the_record_and_others_write_beside_it(tmp_path):
    assert Run(tmp_path).output_dir == tmp_path.resolve() / "outputs"
    assert Run(tmp_path, profile="smoke").output_dir == tmp_path.resolve() / "outputs_smoke"
    assert Run(tmp_path, profile="full_spec+smoke").output_dir.name == "outputs_full_spec_smoke"
    assert Run(tmp_path).is_record and not Run(tmp_path, profile="smoke").is_record


def test_record_writes_tables_and_metadata_without_a_timestamp(tmp_path):
    run = Run(tmp_path, seed=42, config={"n": 10})
    record(run, {"metrics": pd.DataFrame({"auc": [0.7]})})
    first = (run.output_dir / "run_meta.json").read_text()
    record(run, {"metrics": pd.DataFrame({"auc": [0.7]})})
    assert (run.output_dir / "run_meta.json").read_text() == first

    meta = json.loads(first)
    assert meta["profile"] == "default" and meta["seed"] == 42 and meta["config"] == {"n": 10}
    assert "numpy" in meta["packages"]
    assert pd.read_csv(run.output_dir / "metrics.csv")["auc"].tolist() == [0.7]


def test_a_folder_from_one_profile_refuses_another(tmp_path):
    record(Run(tmp_path))
    (tmp_path / "outputs" / "run_meta.json").write_text(json.dumps({"profile": "smoke"}))
    with pytest.raises(ProfileMismatch, match="smoke"):
        Run(tmp_path)


def test_report_figures_are_those_markdown_embeds_names_or_globs(tmp_path):
    out = tmp_path / "outputs"
    (out / "plots").mkdir(parents=True)
    for name in ["embedded.png", "plots/nested.png", "named.svg", "fig_heatmap_auc.png", "orphan.png"]:
        (out / name).write_bytes(b"")
    (tmp_path / "report.md").write_text(
        "![a](outputs/embedded.png)\n![b](outputs/plots/nested.png)\n"
        "- `named.svg` lists PCA views\n- `fig_heatmap_*.png` heatmaps\n"
    )
    found = {p.relative_to(out).as_posix() for p in report_figures(tmp_path)}
    assert found == {"embedded.png", "plots/nested.png", "named.svg", "fig_heatmap_auc.png"}


def test_markdown_in_non_default_outputs_does_not_count(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "fig.png").write_bytes(b"")
    (tmp_path / "outputs_smoke").mkdir()
    (tmp_path / "outputs_smoke" / "report.md").write_text("![x](../outputs/fig.png)")
    assert report_figures(tmp_path) == []


def test_a_run_without_config_records_the_folders_yaml_files(tmp_path):
    (tmp_path / "config.yaml").write_text("seed: 7\nn_train: 100\n")
    (tmp_path / "study.yaml").write_text("p_pos: [0.1, 0.2]\n")
    run = Run(tmp_path)
    assert run.seed == 7
    assert run.config == {"config.yaml": {"seed": 7, "n_train": 100}, "study.yaml": {"p_pos": [0.1, 0.2]}}
