"""Load the 9 Olist CSV files into a local SQLite database (olist.db)."""
import sqlite3
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DB = Path(__file__).resolve().parents[1] / "olist.db"

TABLES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_orders_customer ON orders(customer_id)",
    "CREATE INDEX IF NOT EXISTS ix_items_order ON order_items(order_id)",
    "CREATE INDEX IF NOT EXISTS ix_items_product ON order_items(product_id)",
    "CREATE INDEX IF NOT EXISTS ix_items_seller ON order_items(seller_id)",
    "CREATE INDEX IF NOT EXISTS ix_payments_order ON payments(order_id)",
    "CREATE INDEX IF NOT EXISTS ix_reviews_order ON reviews(order_id)",
    # One coordinate per zip-code prefix (the raw table has ~1M rows, many per prefix).
    "DROP TABLE IF EXISTS zip_coords",
    """CREATE TABLE zip_coords AS
       SELECT geolocation_zip_code_prefix AS zip_prefix,
              AVG(geolocation_lat) AS lat, AVG(geolocation_lng) AS lng
       FROM geolocation GROUP BY geolocation_zip_code_prefix""",
    "CREATE INDEX IF NOT EXISTS ix_zip ON zip_coords(zip_prefix)",
]

def main():
    con = sqlite3.connect(DB)
    for table, file in TABLES.items():
        df = pd.read_csv(DATA / file, encoding="utf-8-sig")
        df.to_sql(table, con, if_exists="replace", index=False)
        print(f"{table:22s} {len(df):>8,} rows")
    for stmt in INDEXES:
        con.execute(stmt)
    con.commit()
    con.close()
    print(f"Database written to {DB}")

if __name__ == "__main__":
    main()
