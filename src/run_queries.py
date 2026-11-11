"""Run every query in sql/business_questions.sql and save each result to results/<name>.csv."""
import re
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQL_FILE = ROOT / "sql" / "business_questions.sql"
RESULTS = ROOT / "results"


def split_queries(text):
    """Return (name, sql) pairs, using the '-- name:' marker lines as separators."""
    parts = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.M)
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts), 2)]


def main():
    RESULTS.mkdir(exist_ok=True)
    con = sqlite3.connect(ROOT / "olist.db")
    pd.set_option("display.width", 140)
    for name, sql in split_queries(SQL_FILE.read_text()):
        df = pd.read_sql_query(sql, con)
        df.to_csv(RESULTS / f"{name}.csv", index=False)
        print(f"\n=== {name} ({len(df)} rows)")
        print(df.to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
