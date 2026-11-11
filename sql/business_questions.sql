-- Business questions on the Olist e-commerce database (SQLite).
-- Each query starts with a "-- name:" line; src/run_queries.py runs them and saves results/<name>.csv.
-- "Late" = delivered after the estimated (promised) delivery date.

-- name: q01_monthly_revenue_growth
-- Monthly orders, revenue and month-over-month growth (CTE + LAG window function).
WITH monthly AS (
    SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
           COUNT(DISTINCT o.order_id)                   AS orders,
           ROUND(SUM(i.price), 0)                       AS revenue
    FROM orders o
    JOIN order_items i ON i.order_id = o.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
      AND o.order_purchase_timestamp >= '2017-01-01'
      AND o.order_purchase_timestamp <  '2018-09-01'
    GROUP BY month
)
SELECT month, orders, revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1)            AS mom_growth_pct,
       SUM(revenue) OVER (ORDER BY month)                        AS cumulative_revenue
FROM monthly
ORDER BY month;

-- name: q02_top_categories_share
-- Top 10 product categories by revenue and their share of total revenue (window SUM over all rows).
WITH cat AS (
    SELECT COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
           SUM(i.price) AS revenue,
           COUNT(DISTINCT i.order_id) AS orders
    FROM order_items i
    JOIN orders o   ON o.order_id = i.order_id
    LEFT JOIN products p ON p.product_id = i.product_id
    LEFT JOIN category_translation t ON t.product_category_name = p.product_category_name
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY category
)
SELECT category, orders, ROUND(revenue, 0) AS revenue,
       ROUND(100.0 * revenue / SUM(revenue) OVER (), 1) AS revenue_share_pct
FROM cat
ORDER BY revenue DESC
LIMIT 10;

-- name: q03_late_rate_by_state
-- Late-delivery rate, actual vs promised delivery days by customer state (CASE + GROUP BY + HAVING).
SELECT c.customer_state,
       COUNT(*) AS delivered_orders,
       ROUND(100.0 * AVG(CASE WHEN date(o.order_delivered_customer_date) > date(o.order_estimated_delivery_date)
                              THEN 1 ELSE 0 END), 1) AS late_rate_pct,
       ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)), 1) AS avg_actual_days,
       ROUND(AVG(julianday(o.order_estimated_delivery_date) - julianday(o.order_purchase_timestamp)), 1) AS avg_promised_days
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
GROUP BY c.customer_state
HAVING COUNT(*) >= 1000
ORDER BY late_rate_pct DESC;

-- name: q04_review_score_late_vs_ontime
-- Does lateness hurt customer satisfaction? Average review score and share of 1-star reviews.
WITH delivered AS (
    SELECT o.order_id,
           CASE WHEN date(o.order_delivered_customer_date) > date(o.order_estimated_delivery_date)
                THEN 'late' ELSE 'on_time' END AS delivery_status
    FROM orders o
    WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
)
SELECT d.delivery_status,
       COUNT(*) AS reviews,
       ROUND(AVG(r.review_score), 2) AS avg_review_score,
       ROUND(100.0 * AVG(CASE WHEN r.review_score = 1 THEN 1 ELSE 0 END), 1) AS one_star_pct
FROM delivered d
JOIN reviews r ON r.order_id = d.order_id
GROUP BY d.delivery_status;

-- name: q05_late_rate_same_vs_cross_state
-- Orders shipped from a seller in another state vs the same state (subquery aggregation + CASE).
WITH order_route AS (
    SELECT o.order_id,
           MAX(CASE WHEN s.seller_state <> c.customer_state THEN 1 ELSE 0 END) AS cross_state,
           MAX(CASE WHEN date(o.order_delivered_customer_date) > date(o.order_estimated_delivery_date)
                    THEN 1 ELSE 0 END) AS is_late
    FROM orders o
    JOIN customers c    ON c.customer_id = o.customer_id
    JOIN order_items i  ON i.order_id = o.order_id
    JOIN sellers s      ON s.seller_id = i.seller_id
    WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY o.order_id
)
SELECT CASE cross_state WHEN 1 THEN 'cross_state' ELSE 'same_state' END AS route,
       COUNT(*) AS orders,
       ROUND(100.0 * AVG(is_late), 1) AS late_rate_pct
FROM order_route
GROUP BY route;

-- name: q06_top3_sellers_per_state
-- Top 3 sellers by revenue within each of the three biggest seller states (DENSE_RANK + PARTITION BY).
WITH seller_rev AS (
    SELECT s.seller_state, i.seller_id,
           ROUND(SUM(i.price), 0) AS revenue,
           COUNT(DISTINCT i.order_id) AS orders
    FROM order_items i
    JOIN sellers s ON s.seller_id = i.seller_id
    JOIN orders o  ON o.order_id = i.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY s.seller_state, i.seller_id
), ranked AS (
    SELECT *, DENSE_RANK() OVER (PARTITION BY seller_state ORDER BY revenue DESC) AS rnk
    FROM seller_rev
)
SELECT seller_state, rnk, substr(seller_id, 1, 8) AS seller_id_short, revenue, orders
FROM ranked
WHERE rnk <= 3 AND seller_state IN ('SP', 'MG', 'PR')
ORDER BY seller_state, rnk;

-- name: q07_repeat_customer_rate
-- Share of real customers (customer_unique_id) who ordered more than once.
WITH per_customer AS (
    SELECT c.customer_unique_id, COUNT(DISTINCT o.order_id) AS n_orders
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY c.customer_unique_id
)
SELECT COUNT(*) AS customers,
       SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END) AS repeat_customers,
       ROUND(100.0 * AVG(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END), 2) AS repeat_rate_pct
FROM per_customer;

-- name: q08_payment_mix
-- Payment methods: share of orders, average order payment and instalments.
SELECT payment_type,
       COUNT(DISTINCT order_id) AS orders,
       ROUND(100.0 * COUNT(DISTINCT order_id) / (SELECT COUNT(DISTINCT order_id) FROM payments), 1) AS order_share_pct,
       ROUND(AVG(payment_value), 1) AS avg_payment_value,
       ROUND(AVG(payment_installments), 1) AS avg_installments
FROM payments
WHERE payment_type <> 'not_defined'
GROUP BY payment_type
ORDER BY orders DESC;

-- name: q09_delivered_orders_without_review
-- Anti-join: delivered orders that never received a review.
SELECT COUNT(*) AS delivered_orders_without_review
FROM orders o
LEFT JOIN reviews r ON r.order_id = o.order_id
WHERE o.order_status = 'delivered' AND r.order_id IS NULL;

-- name: q10_late_rate_by_month
-- Monthly late-delivery rate (seasonality / operational stress), purchase month 2017-01 to 2018-08.
SELECT strftime('%Y-%m', order_purchase_timestamp) AS month,
       COUNT(*) AS delivered_orders,
       ROUND(100.0 * AVG(CASE WHEN date(order_delivered_customer_date) > date(order_estimated_delivery_date)
                              THEN 1 ELSE 0 END), 1) AS late_rate_pct
FROM orders
WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
  AND order_purchase_timestamp >= '2017-01-01' AND order_purchase_timestamp < '2018-09-01'
GROUP BY month
ORDER BY month;
