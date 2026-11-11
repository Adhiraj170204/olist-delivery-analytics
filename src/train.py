"""Train and evaluate late-delivery classifiers on the SQL-built feature table.

Split is by time (train: Jan 2017 - Mar 2018 purchases, test: Apr - Aug 2018) so the model is
always evaluated on orders placed after everything it learned from.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DATE = "2018-04-01"
FLAG_SHARE = 0.10  # business rule: proactively handle the riskiest 10% of orders

NUMERIC = ["promised_days", "approve_hours", "purchase_month", "purchase_dow", "purchase_hour",
           "cross_state", "distance_km", "seller_past_orders", "seller_past_late_rate",
           "n_items", "n_sellers", "total_price", "total_freight", "freight_ratio",
           "total_weight_g", "total_volume_cm3", "installments", "n_payments"]
PRIOR_STRENGTH = 20  # smoothing: a new seller starts at the overall late rate
CATEGORICAL = ["customer_state", "seller_state", "category", "payment_type"]


def load():
    con = sqlite3.connect(ROOT / "olist.db")
    df = pd.read_sql_query((ROOT / "sql" / "features.sql").read_text(), con)
    con.close()
    df["purchase_ts"] = pd.to_datetime(df["purchase_ts"])
    df["delivered_ts"] = pd.to_datetime(df["delivered_ts"])
    return df


def add_distance(df):
    """Great-circle (haversine) distance between customer and seller zip-code centroids, in km."""
    lat1, lon1, lat2, lon2 = (np.radians(df[c]) for c in ["customer_lat", "customer_lng", "seller_lat", "seller_lng"])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))
    return df


def add_seller_history(df, prior_rate):
    """For each order: how many of this seller's earlier orders had been DELIVERED before this purchase,
    and what share of those were late (smoothed towards prior_rate). Uses only past outcomes -> no leakage."""
    events = (df[["seller_id", "delivered_ts", "is_late"]]
              .sort_values("delivered_ts").copy())
    events["cum_orders"] = events.groupby("seller_id").cumcount() + 1
    events["cum_late"] = events.groupby("seller_id")["is_late"].cumsum()
    queries = df[["order_id", "seller_id", "purchase_ts"]].sort_values("purchase_ts")
    hist = pd.merge_asof(queries, events[["seller_id", "delivered_ts", "cum_orders", "cum_late"]],
                         left_on="purchase_ts", right_on="delivered_ts", by="seller_id",
                         direction="backward", allow_exact_matches=False)
    hist[["cum_orders", "cum_late"]] = hist[["cum_orders", "cum_late"]].fillna(0)
    hist["seller_past_orders"] = hist["cum_orders"]
    hist["seller_past_late_rate"] = ((hist["cum_late"] + PRIOR_STRENGTH * prior_rate)
                                     / (hist["cum_orders"] + PRIOR_STRENGTH))
    return df.merge(hist[["order_id", "seller_past_orders", "seller_past_late_rate"]], on="order_id")


def top_share_metrics(y_true, scores, share=FLAG_SHARE):
    """Precision and recall if we flag the top `share` of orders by predicted risk."""
    k = int(np.ceil(share * len(scores)))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-scores)[:k]] = True
    tp = (flagged & (y_true == 1)).sum()
    return {"flagged": int(k), "precision": tp / k, "recall": tp / y_true.sum()}


def evaluate(name, y, scores):
    res = {"model": name,
           "roc_auc": roc_auc_score(y, scores),
           "pr_auc": average_precision_score(y, scores)}
    res.update({f"top10_{k}": v for k, v in top_share_metrics(y.values, scores).items()})
    return res


def main():
    df = add_distance(load())
    prior_rate = df.loc[df.purchase_ts < SPLIT_DATE, "is_late"].mean()  # from training period only
    df = add_seller_history(df, prior_rate)
    train, test = df[df.purchase_ts < SPLIT_DATE], df[df.purchase_ts >= SPLIT_DATE]
    X_tr, y_tr = train[NUMERIC + CATEGORICAL], train.is_late
    X_te, y_te = test[NUMERIC + CATEGORICAL], test.is_late
    print(f"train {len(train):,} orders (late {y_tr.mean():.1%}) | test {len(test):,} orders (late {y_te.mean():.1%})")

    results = []

    # Baseline rule: flag cross-state orders with the shortest promised delivery windows.
    rule_score = test.cross_state.values * 1000 - test.promised_days.values
    results.append(evaluate("Rule: cross-state + shortest promise", y_te, rule_score))

    pre_lr = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50))]), CATEGORICAL),
    ])
    lr = Pipeline([("pre", pre_lr), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    lr.fit(X_tr, y_tr)
    results.append(evaluate("Logistic Regression", y_te, lr.predict_proba(X_te)[:, 1]))

    pre_xgb = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
    ])
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    xgb = Pipeline([("pre", pre_xgb), ("clf", XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="aucpr", random_state=42, n_jobs=-1))])
    xgb.fit(X_tr, y_tr)
    xgb_scores = xgb.predict_proba(X_te)[:, 1]
    results.append(evaluate("XGBoost", y_te, xgb_scores))

    # For comparison only: the same XGBoost evaluated on a RANDOM split. This looks much better because
    # orders from the same (disrupted) months land in both train and test -> an optimistic, time-leaky estimate.
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split
    a, b = train_test_split(df, test_size=0.3, random_state=42, stratify=df.is_late)
    xgb_rand = clone(xgb).fit(a[NUMERIC + CATEGORICAL], a.is_late)
    r = evaluate("XGBoost, random split (leaky, for contrast)", b.is_late,
                 xgb_rand.predict_proba(b[NUMERIC + CATEGORICAL])[:, 1])
    r["note"] = "not a valid estimate"

    res = pd.DataFrame(results + [r])
    res.insert(1, "test_late_rate", [y_te.mean()] * (len(res) - 1) + [b.is_late.mean()])
    print(res.round(3).to_string(index=False))
    res.round(4).to_csv(ROOT / "results" / "model_comparison.csv", index=False)

    # Feature importance (gain) for the XGBoost model, mapped back to feature names.
    names = xgb.named_steps["pre"].get_feature_names_out()
    booster = xgb.named_steps["clf"].get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = pd.Series({names[int(k[1:])]: v for k, v in gain.items()}).sort_values(ascending=False)
    imp = (imp / imp.sum()).head(10).round(3)
    print("\nTop XGBoost features (share of total gain):\n" + imp.to_string())
    imp.to_csv(ROOT / "results" / "xgb_feature_importance.csv", header=["gain_share"])

    with open(ROOT / "results" / "split_info.json", "w") as f:
        json.dump({"split_date": SPLIT_DATE, "train_orders": len(train), "test_orders": len(test),
                   "train_late_rate": round(float(y_tr.mean()), 4),
                   "test_late_rate": round(float(y_te.mean()), 4)}, f, indent=2)


if __name__ == "__main__":
    main()
