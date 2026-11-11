-- One row per delivered order, using only information available at purchase / approval time.
-- Target: is_late = 1 if the order arrived after the promised (estimated) delivery date.
-- Nothing from the carrier or delivery timestamps is used as a feature (that would leak the target).
-- delivered_ts is returned only so train.py can build the seller's PAST late rate without leakage.
WITH items AS (
    SELECT i.order_id,
           COUNT(*)                                   AS n_items,
           COUNT(DISTINCT i.seller_id)                AS n_sellers,
           SUM(i.price)                               AS total_price,
           SUM(i.freight_value)                       AS total_freight,
           SUM(COALESCE(p.product_weight_g, 0))       AS total_weight_g,
           SUM(COALESCE(p.product_length_cm * p.product_height_cm * p.product_width_cm, 0)) AS total_volume_cm3,
           MIN(s.seller_state)                        AS seller_state,
           MIN(COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')) AS category
    FROM order_items i
    JOIN sellers s ON s.seller_id = i.seller_id
    LEFT JOIN products p ON p.product_id = i.product_id
    LEFT JOIN category_translation t ON t.product_category_name = p.product_category_name
    GROUP BY i.order_id
), pay AS (
    SELECT order_id,
           MAX(payment_installments) AS installments,
           COUNT(*)                  AS n_payments,
           -- main payment type = the one with the largest value in the order
           (SELECT p2.payment_type FROM payments p2
             WHERE p2.order_id = p1.order_id
             ORDER BY p2.payment_value DESC LIMIT 1) AS payment_type
    FROM payments p1
    GROUP BY order_id
), main_seller AS (
    -- the order's first seller (most orders have exactly one)
    SELECT order_id, MIN(seller_id) AS seller_id
    FROM order_items
    GROUP BY order_id
), seller_route AS (
    SELECT i.order_id,
           MAX(CASE WHEN s.seller_state <> c.customer_state THEN 1 ELSE 0 END) AS cross_state
    FROM order_items i
    JOIN sellers s   ON s.seller_id = i.seller_id
    JOIN orders o    ON o.order_id = i.order_id
    JOIN customers c ON c.customer_id = o.customer_id
    GROUP BY i.order_id
)
SELECT o.order_id,
       o.order_purchase_timestamp                                                  AS purchase_ts,
       ROUND(julianday(o.order_estimated_delivery_date) - julianday(o.order_purchase_timestamp), 2) AS promised_days,
       ROUND(24 * (julianday(o.order_approved_at) - julianday(o.order_purchase_timestamp)), 2)       AS approve_hours,
       CAST(strftime('%m', o.order_purchase_timestamp) AS INTEGER)                 AS purchase_month,
       CAST(strftime('%w', o.order_purchase_timestamp) AS INTEGER)                 AS purchase_dow,
       CAST(strftime('%H', o.order_purchase_timestamp) AS INTEGER)                 AS purchase_hour,
       c.customer_state,
       it.seller_state,
       ms.seller_id,
       cz.lat AS customer_lat, cz.lng AS customer_lng,
       sz.lat AS seller_lat,   sz.lng AS seller_lng,
       o.order_delivered_customer_date                                             AS delivered_ts,
       r.cross_state,
       it.category,
       it.n_items, it.n_sellers,
       ROUND(it.total_price, 2)   AS total_price,
       ROUND(it.total_freight, 2) AS total_freight,
       ROUND(it.total_freight / NULLIF(it.total_price, 0), 4) AS freight_ratio,
       it.total_weight_g, it.total_volume_cm3,
       pay.payment_type, pay.installments, pay.n_payments,
       CASE WHEN date(o.order_delivered_customer_date) > date(o.order_estimated_delivery_date)
            THEN 1 ELSE 0 END AS is_late
FROM orders o
JOIN customers c      ON c.customer_id = o.customer_id
JOIN items it         ON it.order_id = o.order_id
JOIN seller_route r   ON r.order_id = o.order_id
JOIN main_seller ms   ON ms.order_id = o.order_id
JOIN sellers sm       ON sm.seller_id = ms.seller_id
LEFT JOIN zip_coords cz ON cz.zip_prefix = c.customer_zip_code_prefix
LEFT JOIN zip_coords sz ON sz.zip_prefix = sm.seller_zip_code_prefix
LEFT JOIN pay         ON pay.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_purchase_timestamp >= '2017-01-01'
  AND o.order_purchase_timestamp <  '2018-09-01';
