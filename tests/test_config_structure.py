from pathlib import Path
import yaml


def test_config_has_required_sections():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    for key in ["regions", "features", "paths", "files", "scripts"]:
        assert key in cfg


def test_regions_nonempty():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    assert isinstance(cfg["regions"], list)
    assert len(cfg["regions"]) >= 1
