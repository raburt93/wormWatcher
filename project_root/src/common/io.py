from __future__ import annotations
import json
import pathlib
import datetime as dt
from typing import Any, Dict, Iterable
import pandas as pd


def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def now_iso() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def write_table(df: pd.DataFrame, base: pathlib.Path, csv_mirror: bool) -> None:
    ensure_dir(base.parent)
    df.to_parquet(str(base.with_suffix(".parquet")), index=False)
    if csv_mirror:
        df.to_csv(str(base.with_suffix(".csv")), index=False)
