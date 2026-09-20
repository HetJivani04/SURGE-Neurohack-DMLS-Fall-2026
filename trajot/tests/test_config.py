import copy
import json
import shutil
from pathlib import Path

import psutil
import pytest
import yaml

from trajot.config import Config, ConfigError, config_hash, load_config, resolve_n_jobs, validate_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENTS = ["00_noalign", "01_brainsync", "02_fugw", "03_conn_srm", "10_ours_full", "11_ours_ablated"]


@pytest.fixture
def project(tmp_path):
    """A copy of configs/ with a paths.yaml, so tests never depend on the developer's own."""
    shutil.copytree(CONFIGS, tmp_path / "configs", ignore=shutil.ignore_patterns("paths.yaml"))
    (tmp_path / "configs" / "paths.yaml").write_text(
        f"data_root: {tmp_path / 'data'}\ncontract_version: '1'\n")
    return tmp_path


@pytest.fixture
def exp_path(project):
    return lambda name="00_noalign": project / "configs" / "experiments" / f"{name}.yaml"


def valid_cfg():
    return {
        "experiment": "x",
        "data": {"root": "/somewhere", "contract_version": "1"},
        "run": {"seed": 0, "n_jobs": 1},
        "eval": {"pairs": {"n": 500, "seed": 1}, "permutations": {"B": 10000}},
    }


# ---- load_config -----------------------------------------------------------
@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_experiment_config_is_valid_and_hashable(exp_path, name):
    cfg = load_config(exp_path(name))
    assert cfg.experiment == name
    assert len(cfg.hash) == 64
    assert cfg.get("run.n_jobs") == 1
    assert cfg.get("eval.pairs.n") == 500 and cfg.get("eval.permutations.B") == 10000
    assert cfg.get("data.contract_version") == "1"
    json.dumps(cfg.raw)


def test_model_and_eval_configs_are_merged_in(exp_path):
    cfg = load_config(exp_path("10_ours_full"))
    assert (cfg.get("model.K"), cfg.get("model.d"), cfg.get("model.r")) == (512, 32, 32)
    assert (cfg.get("model.m_draws"), cfg.get("model.batch_subjects")) == (4, 8)
    assert cfg.get("model.sinkhorn.L") == 30 and cfg.get("model.sinkhorn.eps") == [0.1, 0.01]
    assert cfg.get("model.beta.warmup_frac") == 0.3
    assert cfg.get("model.gauge_features") is None
    assert cfg.get("eval.pairs.n") == 500


def test_gauge_features_false_only_in_the_ablation(exp_path):
    full = load_config(exp_path("10_ours_full")).raw["model"]
    ablated = load_config(exp_path("11_ours_ablated")).raw["model"]
    assert ablated.pop("gauge_features") is False
    assert ablated == full


def test_missing_paths_yaml_says_how_to_fix_it(project):
    (project / "configs" / "paths.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="copy configs/paths.example.yaml to configs/paths.yaml"):
        load_config(project / "configs" / "experiments" / "00_noalign.yaml")


def test_paths_yaml_keys_land_under_data(exp_path, project):
    cfg = load_config(exp_path())
    assert cfg.data_root == project / "data"
    assert cfg.get("data.contract_version") == "1"


def test_overrides_are_applied_last(exp_path):
    cfg = load_config(exp_path(), {"run.seed": 7, "model.sinkhorn.L": 5, "eval.pairs.seed": 9})
    assert (cfg.get("run.seed"), cfg.get("model.sinkhorn.L"), cfg.get("eval.pairs.seed")) == (7, 5, 9)


# ---- Config ----------------------------------------------------------------
def test_config_is_a_mapping_with_dotted_get(exp_path):
    cfg = load_config(exp_path())
    assert isinstance(cfg, Config)
    assert cfg["run"]["seed"] == 0 and cfg.get("run.seed") == 0
    assert cfg.get("run.nope", "dflt") == "dflt" and cfg.get("nope.nope") is None
    assert "run" in cfg and len(cfg) == len(cfg.raw)
    assert cfg.n_jobs == 1


# ---- config_hash -----------------------------------------------------------
def test_hash_ignores_filename_key_order_and_data_root(project):
    exp = project / "configs" / "experiments"
    original = load_config(exp / "00_noalign.yaml")

    data = yaml.safe_load((exp / "00_noalign.yaml").read_text())
    (exp / "renamed_copy.yaml").write_text(yaml.safe_dump(dict(reversed(list(data.items()))), sort_keys=False))
    assert load_config(exp / "renamed_copy.yaml").hash == original.hash

    (project / "configs" / "paths.yaml").write_text("data_root: /other/machine/ds000243\ncontract_version: '1'\n")
    other = load_config(exp / "00_noalign.yaml")
    assert other.data_root != original.data_root and other.hash == original.hash


@pytest.mark.parametrize("key,value", [
    ("run.seed", 1), ("run.n_jobs", 2), ("eval.pairs.seed", 3), ("model.K", 256),
    ("model.beta.warmup_frac", 0.5), ("run.debug", True),
])
def test_any_single_override_changes_the_hash(exp_path, key, value):
    assert load_config(exp_path(), {key: value}).hash != load_config(exp_path()).hash


def test_hash_tracks_contract_version(project):
    path = project / "configs" / "experiments" / "00_noalign.yaml"
    before = load_config(path)
    (project / "configs" / "paths.yaml").write_text(f"data_root: {project}\ncontract_version: '2'\n")
    assert load_config(path).hash != before.hash


def test_config_hash_is_deterministic_and_order_independent():
    a = valid_cfg()
    assert config_hash(a) == config_hash(valid_cfg())
    assert config_hash(a) == config_hash(dict(reversed(list(copy.deepcopy(a).items()))))
    a["run"]["seed"] = 1
    assert config_hash(a) != config_hash(valid_cfg())


# ---- validate_config -------------------------------------------------------
def test_valid_config_passes():
    validate_config(valid_cfg())


@pytest.mark.parametrize("path", [
    "experiment", "data.contract_version", "data.root", "run.seed", "run.n_jobs",
    "eval.pairs.n", "eval.pairs.seed", "eval.permutations.B",
])
def test_missing_required_key_names_the_path(path):
    cfg = valid_cfg()
    *parents, leaf = path.split(".")
    node = cfg
    for p in parents:
        node = node[p]
    del node[leaf]
    with pytest.raises(ConfigError, match=path.replace(".", r"\.")):
        validate_config(cfg)


@pytest.mark.parametrize("path,value,expected", [
    ("run.seed", "zero", "int"), ("run.seed", 1.5, "int"), ("data.root", 3, "str"),
    ("data.contract_version", 1, "str"), ("eval.pairs.seed", "a", "int"),
])
def test_wrong_type_reports_key_expected_type_and_value(path, value, expected):
    cfg = valid_cfg()
    *parents, leaf = path.split(".")
    node = cfg
    for p in parents:
        node = node[p]
    node[leaf] = value
    with pytest.raises(ConfigError) as err:
        validate_config(cfg)
    assert path in str(err.value) and f"expected {expected}" in str(err.value) and repr(value) in str(err.value)


def test_n_jobs_may_be_null():
    cfg = valid_cfg()
    cfg["run"]["n_jobs"] = None
    validate_config(cfg)


@pytest.mark.parametrize("section,leaf,value", [("pairs", "n", 100), ("permutations", "B", 100)])
def test_pair_and_permutation_counts_are_pinned_unless_debug(section, leaf, value):
    cfg = valid_cfg()
    cfg["eval"][section][leaf] = value
    with pytest.raises(ConfigError, match="run.debug"):
        validate_config(cfg)
    cfg["run"]["debug"] = True
    validate_config(cfg)


# ---- resolve_n_jobs --------------------------------------------------------
def test_resolve_n_jobs():
    assert resolve_n_jobs({"run": {"n_jobs": 3}}) == 3
    assert resolve_n_jobs({"run": {"n_jobs": None}}) == psutil.cpu_count(logical=False)
    assert resolve_n_jobs({"run": {}}) == psutil.cpu_count(logical=False)
