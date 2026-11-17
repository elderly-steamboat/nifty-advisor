import shutil
from pathlib import Path

import pytest
import yaml

from advisor.config import load_config

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


@pytest.fixture
def tmp_cfg(tmp_path):
    """Repo config with data_dir pointed at an empty temp dir."""
    data = yaml.safe_load((ROOT / "config.yaml").read_text())
    data["data_dir"] = str(tmp_path / "data")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return load_config(p)


@pytest.fixture
def seeded_cfg(tmp_cfg):
    """tmp_cfg with the real 2026-11-17 and 2023-06-15 fixture files placed in the cache."""
    from advisor.data import ingest_index_file
    d = tmp_cfg.data_dir
    (d / "bhavcopy").mkdir(parents=True)
    shutil.copy(FIX / "udiff_2026-11-17.csv.zip", d / "bhavcopy" / "2026-11-17.csv.zip")
    shutil.copy(FIX / "legacy_2023-06-15.csv.zip", d / "bhavcopy" / "2023-06-15.csv.zip")
    for day in ("2026-11-17", "2023-06-15"):
        ingest_index_file(FIX / f"ind_close_all_{day}.csv", d)
    return tmp_cfg
