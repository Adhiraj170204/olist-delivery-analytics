# E-commerce Delivery Analytics & Late-Delivery Prediction (SQL + ML)

Late deliveries are the biggest driver of bad reviews on Olist, a Brazilian e-commerce marketplace.
This project uses **SQL** to analyse ~100k real orders (2016–2018) across 9 linked tables, then builds a
**leakage-safe machine-learning model** that flags, at purchase time, the orders most likely to arrive late.

## Key findings

| Finding | Number | Query |
|---|---|---|
| Late orders get far worse reviews | avg **2.27★** late vs **4.29★** on time; **53.7%** of late orders get 1★ (vs 6.6%) | `q04` |
| Overall late-delivery rate | **6.8%** of delivered orders | `q10` |
| Cross-state shipping is riskier | **8.0%** late vs **4.5%** within the same state | `q05` |
| Lateness comes in network-wide spikes | **12.4%** late in Nov 2017 (Black Friday), **14.1%** Feb 2018, **19.0%** Mar 2018 | `q10` |
| Worst states | CE 13.8%, BA 12.2%, RJ 12.1% late (states with 1,000+ orders) | `q03` |
| Customers rarely come back | only **3.0%** of 94,990 customers ordered more than once | `q07` |
| Revenue concentration | top 10 categories = **62%** of revenue; health & beauty leads at 9.3% | `q02` |

## SQL (sql/business_questions.sql)

Ten business questions, each saved to `results/<name>.csv`:

1. Monthly revenue, month-over-month growth and running total (`LAG`, `SUM() OVER`)
2. Top 10 categories and their share of revenue (window `SUM() OVER ()`)
3. Late rate, actual vs promised delivery days by state (`CASE`, `GROUP BY`, `HAVING`)
4. Review score for late vs on-time orders (CTE + join)
5. Late rate for cross-state vs same-state shipments (order-level aggregation)
6. Top 3 sellers by revenue within each major state (`DENSE_RANK() OVER (PARTITION BY ...)`)
7. Repeat-customer rate (`customer_unique_id`)
8. Payment-method mix (scalar subquery for share)
9. Delivered orders without a review (anti-join: `LEFT JOIN ... IS NULL`)
10. Monthly late-delivery rate

The model's feature table is also built in SQL (`sql/features.sql`): one row per delivered order with
promise window, approval delay, basket size, price, freight, weight/volume, category, payment,
customer/seller states and coordinates.

## Machine learning (src/train.py)

**Target:** `is_late` = delivered after the promised date (6.8% overall, so the classes are imbalanced).

**Leakage controls**
- Only information available at purchase/approval time is used; carrier and delivery timestamps are excluded.
- *Seller past late rate* is computed only from that seller's orders **delivered before** each purchase
  (`pandas.merge_asof`), smoothed toward the training-period average for new sellers.
- **Time-based split:** train on purchases Jan 2017 – Mar 2018 (64,053 orders, 8.0% late),
  test on Apr – Aug 2018 (32,150 orders, 4.4% late).

**Results on the time-based test set**

| Approach | ROC-AUC | PR-AUC | Top-10% flagged: precision | Top-10% flagged: recall |
|---|---|---|---|---|
| Rule: cross-state + shortest promise | 0.59 | 0.061 | 7.0% | 15.9% |
| Logistic Regression (scaled, one-hot, class weights) | 0.70 | 0.094 | 10.8% | 24.3% |
| **XGBoost** (scale_pos_weight, depth 4) | **0.71** | **0.113** | **12.5%** | **28.3%** |
| XGBoost on a *random* split (for contrast) | 0.79 | 0.261 | 27.1% | 39.9% |

- PR-AUC 0.113 is **2.6×** the 4.4% base rate. Flagging the riskiest 10% of orders catches **28%** of late
  orders, **2.8×** better than flagging at random.
- The random split looks much better only because orders from the same disrupted months fall in both train
  and test, so the model memorises *when* disruptions happened. That is a time leak, so the time-based
  numbers are the honest estimate.

## What it means for the business

- Late delivery is the main lever on customer satisfaction (2.27★ vs 4.29★).
- Much of the lateness comes from **network-wide shocks** (Black Friday, Feb–Mar 2018). Single-feature checks
  show distance and cross-state status carry little signal once the promised date is set, because the promise
  already accounts for distance (correlation 0.52; cross-state orders are promised 27.1 days vs 17.5 same-state). **Short promise windows** are the clearest order-level signal.
- Recommended actions: (1) use the model to send proactive delay messages or pick faster carriers for the
  riskiest ~10% of orders; (2) add capacity monitoring and alerting ahead of peak periods, which a per-order
  model cannot foresee; (3) review promise windows for high-late states such as CE, BA and RJ.

## Run it

```bash
pip install -r requirements.txt
python src/download_data.py     # ~120 MB of CSVs from the official Olist GitHub
python src/load_db.py           # builds olist.db (SQLite) with indexes
python src/run_queries.py       # runs the 10 SQL questions -> results/*.csv
python src/train.py             # builds features in SQL, trains, evaluates -> results/
```

SQLite date functions used here have MySQL equivalents: `strftime('%Y-%m', d)` → `DATE_FORMAT(d, '%Y-%m')`,
`julianday(a) - julianday(b)` → `DATEDIFF(a, b)`.

## Limitations and next steps

- Network-wide disruptions are not predictable from order attributes; adding external signals
  (holiday calendar, regional carrier load, weather) is the natural next step.
- Probabilities are not calibrated; calibrate before using them as risk scores.
- Data: [Olist Brazilian E-Commerce Public Dataset](https://github.com/olist/work-at-olist-data) (CC BY-NC-SA 4.0).
