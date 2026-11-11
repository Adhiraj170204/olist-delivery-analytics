"""Download the public Olist Brazilian e-commerce dataset (official Olist GitHub mirror) into data/."""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"
FILES = [
    "olist_orders_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    DATA.mkdir(exist_ok=True)
    for name in FILES:
        target = DATA / name
        if target.exists():
            print(f"skip   {name} (already downloaded)")
            continue
        print(f"fetch  {name} ...")
        urllib.request.urlretrieve(BASE + name, target)
    print("Done.")


if __name__ == "__main__":
    main()
