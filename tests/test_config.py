from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from advisor.config import config_hash, load_config, load_events, load_holidays

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config.yaml"


def _copy_cfg(tmp_path, mutate):
    data = yaml.safe_load(CFG.read_text())
    mutate(data)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_defaults_load():
    cfg = load_config(CFG)
    assert cfg.edge.r_naked == 1.30
    assert cfg.scoring.weights.kronos_drift == 0.0 and cfg.scoring.weights.oi == 0.40 and cfg.kronos.recenter is True
    assert cfg.exits.time_exit_days == 7
    assert cfg.data_dir.is_absolute()


def test_weights_must_sum_to_one(tmp_path):
    p = _copy_cfg(tmp_path, lambda d: d["scoring"]["weights"].update(skew=0.10))
    with pytest.raises(ValidationError):
        load_config(p)


def test_unknown_key_rejected(tmp_path):
    p = _copy_cfg(tmp_path, lambda d: d["edge"].update(bogus=1))
    with pytest.raises(ValidationError):
        load_config(p)


def test_hash_stable_and_sensitive(tmp_path):
    a = load_config(CFG)
    assert config_hash(a) == config_hash(load_config(CFG))
    assert len(config_hash(a)) == 12
    p = _copy_cfg(tmp_path, lambda d: d["edge"].update(r_good=1.11))
    assert config_hash(load_config(p)) != config_hash(a)


def test_events_and_holidays_parse(tmp_path):
    ev = tmp_path / "events.yaml"
    ev.write_text("- {date: 2026-12-04, name: RBI MPC, kind: rbi}\n")
    hol = tmp_path / "holidays.yaml"
    hol.write_text("2026: [2026-12-25]\n")
    events = load_events(ev)
    assert events[0].date == date(2026, 12, 4) and events[0].kind == "rbi"
    assert load_holidays(hol) == {date(2026, 12, 25)}


def test_repo_events_and_holidays_load():
    assert len(load_events(ROOT / "events.yaml")) > 0
    assert len(load_holidays(ROOT / "holidays.yaml")) > 0
