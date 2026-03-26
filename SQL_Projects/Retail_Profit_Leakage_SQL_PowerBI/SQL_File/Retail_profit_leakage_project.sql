-- Retail Profit Leakage & Delivery Efficiency
-- Goal: find where sales are turning into weak profit or losses.
--
-- Questions this covers:
-- Are we growing profitably or just growing revenue?
-- Which regions, categories and products are driving losses?
-- At what discount level do margins break?
-- Is shipping choice adding pressure in some lanes?
--
-- Table used: superstore_raw
-- Dialect: MySQL 8.0+


-- Sanity checks before doing anything else

SELECT COUNT(*) AS total_rows
FROM superstore_raw;

SELECT
    MIN(LEFT(`Order.Date`, 10)) AS first_order_date,
    MAX(LEFT(`Order.Date`, 10)) AS last_order_date,
    MIN(Discount) AS min_discount,
    MAX(Discount) AS max_discount,
    MIN(Profit) AS min_profit,
    MAX(Profit) AS max_profit
FROM superstore_raw;

SELECT
    SUM(CASE WHEN `Order.ID` IS NULL OR `Order.ID` = '' THEN 1 ELSE 0 END) AS missing_order_id,
    SUM(CASE WHEN `Product.ID` IS NULL OR `Product.ID` = '' THEN 1 ELSE 0 END) AS missing_product_id,
    SUM(CASE WHEN `Customer.ID` IS NULL OR `Customer.ID` = '' THEN 1 ELSE 0 END) AS missing_customer_id,
    SUM(CASE WHEN Sales IS NULL THEN 1 ELSE 0 END) AS missing_sales,
    SUM(CASE WHEN Profit IS NULL THEN 1 ELSE 0 END) AS missing_profit,
    SUM(CASE WHEN Discount IS NULL THEN 1 ELSE 0 END) AS missing_discount,
    SUM(CASE WHEN `Shipping.Cost` IS NULL THEN 1 ELSE 0 END) AS missing_shipping_cost
FROM superstore_raw;

-- No gaps in the key columns on this import. Good to proceed.


-- Clean the raw import once and reuse it everywhere.
-- Also set up a few fields I'll need again and again.

CREATE OR REPLACE VIEW vw_orders_clean AS
SELECT
    `Row.ID` AS row_id,
    `Order.ID` AS order_id,
    STR_TO_DATE(LEFT(`Order.Date`, 10), '%Y-%m-%d') AS order_date,
    STR_TO_DATE(LEFT(`Ship.Date`, 10), '%Y-%m-%d') AS ship_date,

    `Customer.ID` AS customer_id,
    `Customer.Name` AS customer_name,

    `Product.ID` AS product_id,
    `Product.Name` AS product_name,

    Category AS category,
    `Sub.Category` AS sub_category,
    Segment AS segment,

    Market AS market,
    Market2 AS market_group,
    Country AS country,
    Region AS region,
    State AS state,
    City AS city,

    `Order.Priority` AS order_priority,
    `Ship.Mode` AS ship_mode,

    CAST(Sales AS DECIMAL(18,2)) AS sales,
    CAST(Profit AS DECIMAL(18,2)) AS profit,
    CAST(Discount AS DECIMAL(6,4)) AS discount,
    CAST(`Shipping.Cost` AS DECIMAL(18,2)) AS shipping_cost,
    Quantity AS quantity,

    YEAR(STR_TO_DATE(LEFT(`Order.Date`, 10), '%Y-%m-%d')) AS order_year,
    MONTH(STR_TO_DATE(LEFT(`Order.Date`, 10), '%Y-%m-%d')) AS order_month_num,
    DATE_FORMAT(STR_TO_DATE(LEFT(`Order.Date`, 10), '%Y-%m-%d'), '%Y-%m') AS order_month,
    weeknum AS order_week,

    DATEDIFF(
        STR_TO_DATE(LEFT(`Ship.Date`, 10), '%Y-%m-%d'),
        STR_TO_DATE(LEFT(`Order.Date`, 10), '%Y-%m-%d')
    ) AS delivery_days,

    CASE WHEN Profit < 0 THEN 1 ELSE 0 END AS is_loss_order,

    CASE WHEN Profit < 0 THEN ABS(Profit) ELSE 0 END AS loss_amount,

    CASE
        WHEN Sales = 0 THEN NULL
        ELSE Profit / Sales
    END AS profit_margin,

    CASE
        WHEN Sales = 0 THEN NULL
        ELSE `Shipping.Cost` / Sales
    END AS shipping_cost_pct_of_sales,

    CASE
        WHEN Discount = 0 THEN '0%'
        WHEN Discount > 0 AND Discount <= 0.05 THEN '0-5%'
        WHEN Discount > 0.05 AND Discount <= 0.10 THEN '5-10%'
        WHEN Discount > 0.10 AND Discount <= 0.20 THEN '10-20%'
        WHEN Discount > 0.20 AND Discount <= 0.30 THEN '20-30%'
        ELSE '30%+'
    END AS discount_bucket,

    CASE
        WHEN DATEDIFF(
            STR_TO_DATE(LEFT(`Ship.Date`, 10), '%Y-%m-%d'),
            STR_TO_DATE(LEFT(`Order.Date`, 10), '%Y-%m-%d')
        ) <= 2 THEN '0-2 days'
        WHEN DATEDIFF(
            STR_TO_DATE(LEFT(`Ship.Date`, 10), '%Y-%m-%d'),
            STR_TO_DATE(LEFT(`Order.Date`, 10), '%Y-%m-%d')
        ) <= 4 THEN '3-4 days'
        WHEN DATEDIFF(
            STR_TO_DATE(LEFT(`Ship.Date`, 10), '%Y-%m-%d'),
            STR_TO_DATE(LEFT(`Order.Date`, 10), '%Y-%m-%d')
        ) <= 6 THEN '5-6 days'
        ELSE '7+ days'
    END AS delivery_bucket
FROM superstore_raw;

SELECT *
FROM vw_orders_clean
LIMIT 10;

SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN order_date IS NULL THEN 1 ELSE 0 END) AS bad_order_dates,
    SUM(CASE WHEN ship_date IS NULL THEN 1 ELSE 0 END) AS bad_ship_dates,
    SUM(CASE WHEN delivery_days < 0 THEN 1 ELSE 0 END) AS negative_delivery_days,
    SUM(CASE WHEN sales < 0 THEN 1 ELSE 0 END) AS negative_sales,
    SUM(CASE WHEN quantity <= 0 THEN 1 ELSE 0 END) AS zero_or_negative_quantity
FROM vw_orders_clean;


-- Baseline: what the business looks like before drilling down

SELECT
    COUNT(DISTINCT order_id) AS orders,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit,
    ROUND(SUM(profit) / NULLIF(SUM(sales), 0), 4) AS profit_margin,
    ROUND(AVG(discount), 4) AS avg_discount,
    ROUND(SUM(shipping_cost), 2) AS total_shipping_cost,
    ROUND(SUM(shipping_cost) / NULLIF(SUM(sales), 0), 4) AS shipping_cost_pct_of_sales,
    SUM(is_loss_order) AS loss_order_lines,
    ROUND(SUM(loss_amount), 2) AS total_loss_amount
FROM vw_orders_clean;

-- Overall margin is about 11.6%, which looks fine at first glance.
-- But there are roughly 12.5K loss-making lines adding up to about 921K in losses.

CREATE OR REPLACE VIEW vw_executive_kpis AS
SELECT
    COUNT(DISTINCT order_id) AS orders,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(SUM(profit), 2) AS total_profit,
    ROUND(SUM(profit) / NULLIF(SUM(sales), 0), 4) AS profit_margin,
    ROUND(AVG(discount), 4) AS avg_discount,
    ROUND(SUM(shipping_cost), 2) AS total_shipping_cost,
    ROUND(SUM(shipping_cost) / NULLIF(SUM(sales), 0), 4) AS shipping_cost_pct_of_sales,
    SUM(is_loss_order) AS loss_order_lines,
    ROUND(SUM(loss_amount), 2) AS total_loss_amount
FROM vw_orders_clean;


-- Monthly trend: check whether the profit issue is steady or getting worse

CREATE OR REPLACE VIEW vw_monthly_kpis AS
WITH monthly AS
(
    SELECT
        CAST(DATE_FORMAT(order_date, '%Y-%m-01') AS DATE) AS month_start,
        COUNT(DISTINCT order_id) AS orders,
        SUM(sales) AS total_sales,
        SUM(profit) AS total_profit,
        SUM(loss_amount) AS total_loss_amount,
        AVG(discount) AS avg_discount,
        SUM(shipping_cost) AS total_shipping_cost
    FROM vw_orders_clean
    GROUP BY CAST(DATE_FORMAT(order_date, '%Y-%m-01') AS DATE)
)
SELECT
    month_start,
    orders,
    ROUND(total_sales, 2) AS total_sales,
    ROUND(total_profit, 2) AS total_profit,
    ROUND(total_profit / NULLIF(total_sales, 0), 4) AS profit_margin,
    ROUND(total_loss_amount, 2) AS total_loss_amount,
    ROUND(avg_discount, 4) AS avg_discount,
    ROUND(total_shipping_cost, 2) AS total_shipping_cost,
    ROUND(total_sales - LAG(total_sales) OVER (ORDER BY month_start), 2) AS sales_change_vs_prev_month,
    ROUND(total_profit - LAG(total_profit) OVER (ORDER BY month_start), 2) AS profit_change_vs_prev_month,
    ROUND(
        total_profit / NULLIF(total_sales, 0)
        - LAG(total_profit / NULLIF(total_sales, 0)) OVER (ORDER BY month_start),
        4
    ) AS margin_change_vs_prev_month
FROM monthly;

SELECT *
FROM vw_monthly_kpis
ORDER BY month_start;

-- Revenue grows over time, but margin moves around month to month.
-- That usually means the problem is sitting inside a few pockets, not across everything.


-- Yearly trend for a cleaner management view

CREATE OR REPLACE VIEW vw_yearly_kpis AS
WITH yearly AS
(
    SELECT
        order_year,
        COUNT(DISTINCT order_id) AS orders,
        SUM(sales) AS total_sales,
        SUM(profit) AS total_profit,
        SUM(is_loss_order) AS loss_order_lines,
        SUM(loss_amount) AS loss_amount
    FROM vw_orders_clean
    GROUP BY order_year
)
SELECT
    order_year,
    orders,
    ROUND(total_sales, 2) AS total_sales,
    ROUND(total_profit, 2) AS total_profit,
    ROUND(total_profit / NULLIF(total_sales, 0), 4) AS profit_margin,
    loss_order_lines,
    ROUND(loss_amount, 2) AS loss_amount,
    ROUND(total_sales - LAG(total_sales) OVER (ORDER BY order_year), 2) AS sales_yoy_change,
    ROUND(total_profit - LAG(total_profit) OVER (ORDER BY order_year), 2) AS profit_yoy_change
FROM yearly;

SELECT *
FROM vw_yearly_kpis
ORDER BY order_year;

-- Revenue climbs each year.
-- Margin stays in a tight 11-12% band, so the real story is where profit is leaking.


-- Now find where the losses sit. Start broad, then narrow down.

SELECT
    region,
    COUNT(DISTINCT order_id) AS orders,
    ROUND(SUM(sales), 2) AS sales,
    ROUND(SUM(profit), 2) AS profit,
    ROUND(SUM(loss_amount), 2) AS loss_amount,
    ROUND(SUM(profit) / NULLIF(SUM(sales), 0), 4) AS profit_margin
FROM vw_orders_clean
GROUP BY region
ORDER BY loss_amount DESC, profit ASC;

-- Central has the biggest loss pool.
-- South and EMEA also deserve attention.

SELECT
    category,
    sub_category,
    COUNT(DISTINCT order_id) AS orders,
    ROUND(SUM(sales), 2) AS sales,
    ROUND(SUM(profit), 2) AS profit,
    ROUND(SUM(loss_amount), 2) AS loss_amount,
    ROUND(AVG(discount), 4) AS avg_discount,
    ROUND(AVG(delivery_days), 2) AS avg_delivery_days,
    ROUND(SUM(profit) / NULLIF(SUM(sales), 0), 4) AS profit_margin
FROM vw_orders_clean
GROUP BY category, sub_category
ORDER BY loss_amount DESC, profit ASC;

-- Tables stands out right away: strong sales, but negative margin.
-- Bookcases, Phones, Chairs and Machines also carry big loss pools.

SELECT
    product_id,
    product_name,
    category,
    sub_category,
    COUNT(*) AS order_lines,
    ROUND(SUM(sales), 2) AS sales,
    ROUND(SUM(profit), 2) AS profit,
    ROUND(SUM(loss_amount), 2) AS loss_amount,
    ROUND(AVG(discount), 4) AS avg_discount,
    ROUND(SUM(profit) / NULLIF(SUM(sales), 0), 4) AS profit_margin
FROM vw_orders_clean
GROUP BY product_id, product_name, category, sub_category
HAVING SUM(loss_amount) > 0
ORDER BY loss_amount DESC
LIMIT 25;


-- Pareto check: are losses spread out, or concentrated in a few areas?

CREATE OR REPLACE VIEW vw_subcategory_pareto AS
WITH subcat_loss AS
(
    SELECT
        category,
        sub_category,
        ROUND(SUM(loss_amount), 2) AS loss_amount,
        ROUND(SUM(sales), 2) AS sales,
        ROUND(SUM(profit), 2) AS profit
    FROM vw_orders_clean
    GROUP BY category, sub_category
    HAVING SUM(loss_amount) > 0
),
ranked_loss AS
(
    SELECT
        category,
        sub_category,
        loss_amount,
        sales,
        profit,
        SUM(loss_amount) OVER (ORDER BY loss_amount DESC, sub_category) AS running_loss_amount,
        SUM(loss_amount) OVER () AS total_loss_amount,
        DENSE_RANK() OVER (ORDER BY loss_amount DESC) AS loss_rank
    FROM subcat_loss
)
SELECT
    category,
    sub_category,
    sales,
    profit,
    loss_amount,
    loss_rank,
    ROUND(running_loss_amount, 2) AS running_loss_amount,
    ROUND(total_loss_amount, 2) AS total_loss_amount,
    ROUND(running_loss_amount / NULLIF(total_loss_amount, 0), 4) AS cumulative_loss_pct
FROM ranked_loss;

SELECT *
FROM vw_subcategory_pareto
ORDER BY loss_rank, sub_category;

-- The top 5 sub-categories drive a little over 56% of total losses.
-- That makes the fix list much shorter than it first looks.


-- Heavy discounting looks like a likely driver, so check where margins break

CREATE OR REPLACE VIEW vw_discount_guardrails AS
SELECT
    category,
    sub_category,
    discount_bucket,
    COUNT(*) AS order_lines,
    COUNT(DISTINCT order_id) AS orders,
    ROUND(SUM(sales), 2) AS sales,
    ROUND(SUM(profit), 2) AS profit,
    ROUND(SUM(loss_amount), 2) AS loss_amount,
    ROUND(AVG(discount), 4) AS avg_discount,
    ROUND(SUM(profit) / NULLIF(SUM(sales), 0), 4) AS profit_margin,
    ROUND(AVG(delivery_days), 2) AS avg_delivery_days
FROM vw_orders_clean
GROUP BY category, sub_category, discount_bucket;

SELECT *
FROM vw_discount_guardrails
ORDER BY category, sub_category,
    CASE discount_bucket
        WHEN '0%' THEN 1
        WHEN '0-5%' THEN 2
        WHEN '5-10%' THEN 3
        WHEN '10-20%' THEN 4
        WHEN '20-30%' THEN 5
        ELSE 6
    END;

SELECT
    category,
    sub_category,
    discount_bucket,
    sales,
    profit,
    loss_amount,
    profit_margin
FROM vw_discount_guardrails
WHERE profit_margin < 0
ORDER BY loss_amount DESC, sales DESC;

-- The 30%+ bucket is where margins clearly break.
-- 20-30% is already shaky for some furniture lines, especially Tables and Bookcases.


-- Shipping: check whether mode choice is adding pressure in some lanes

CREATE OR REPLACE VIEW vw_shipping_diagnostics AS
SELECT
    ship_mode,
    delivery_bucket,
    region,
    COUNT(DISTINCT order_id) AS orders,
    ROUND(SUM(sales), 2) AS sales,
    ROUND(SUM(profit), 2) AS profit,
    ROUND(SUM(loss_amount), 2) AS loss_amount,
    ROUND(AVG(delivery_days), 2) AS avg_delivery_days,
    ROUND(SUM(shipping_cost), 2) AS total_shipping_cost,
    ROUND(SUM(shipping_cost) / NULLIF(SUM(sales), 0), 4) AS shipping_cost_pct_of_sales,
    ROUND(SUM(profit) / NULLIF(SUM(sales), 0), 4) AS profit_margin
FROM vw_orders_clean
GROUP BY ship_mode, delivery_bucket, region;

SELECT *
FROM vw_shipping_diagnostics
ORDER BY loss_amount DESC, shipping_cost_pct_of_sales DESC;

-- Shipping is a secondary issue here, not the main one.
-- Standard Class still shows the biggest loss pools because that volume is large, especially in Central and South.


-- Useful BI cut: check whether segment and priority behave differently

CREATE OR REPLACE VIEW vw_segment_priority_diagnostics AS
SELECT
    segment,
    order_priority,
    region,
    COUNT(DISTINCT order_id) AS orders,
    ROUND(SUM(sales), 2) AS sales,
    ROUND(SUM(profit), 2) AS profit,
    ROUND(SUM(loss_amount), 2) AS loss_amount,
    ROUND(AVG(discount), 4) AS avg_discount,
    ROUND(AVG(delivery_days), 2) AS avg_delivery_days,
    ROUND(SUM(profit) / NULLIF(SUM(sales), 0), 4) AS profit_margin
FROM vw_orders_clean
GROUP BY segment, order_priority, region;

SELECT *
FROM vw_segment_priority_diagnostics
ORDER BY loss_amount DESC, avg_discount DESC;


-- Build product risk profiles, then flag the orders tied to them

CREATE OR REPLACE VIEW vw_product_profit_profile AS
SELECT
    product_id,
    product_name,
    category,
    sub_category,
    COUNT(*) AS order_lines,
    COUNT(DISTINCT order_id) AS orders,
    ROUND(SUM(sales), 2) AS product_sales,
    ROUND(SUM(profit), 2) AS product_profit,
    ROUND(SUM(loss_amount), 2) AS product_loss_amount,
    ROUND(AVG(discount), 4) AS avg_discount,
    ROUND(AVG(delivery_days), 2) AS avg_delivery_days,
    ROUND(SUM(profit) / NULLIF(SUM(sales), 0), 4) AS product_margin
FROM vw_orders_clean
GROUP BY product_id, product_name, category, sub_category;

CREATE OR REPLACE VIEW vw_loss_alerts AS
SELECT
    o.region,
    o.segment,
    o.ship_mode,
    o.product_id,
    p.product_name,
    p.category,
    p.sub_category,
    COUNT(*) AS flagged_order_lines,
    COUNT(DISTINCT o.order_id) AS flagged_orders,
    ROUND(SUM(o.sales), 2) AS sales,
    ROUND(SUM(o.profit), 2) AS profit,
    ROUND(SUM(o.loss_amount), 2) AS loss_amount,
    ROUND(AVG(o.discount), 4) AS avg_discount,
    ROUND(AVG(o.delivery_days), 2) AS avg_delivery_days,
    p.product_sales,
    p.product_profit,
    p.product_loss_amount,
    p.product_margin
FROM vw_orders_clean o
JOIN vw_product_profit_profile p
    ON o.product_id = p.product_id
WHERE o.is_loss_order = 1
   OR p.product_margin < 0.05
GROUP BY
    o.region,
    o.segment,
    o.ship_mode,
    o.product_id,
    p.product_name,
    p.category,
    p.sub_category,
    p.product_sales,
    p.product_profit,
    p.product_loss_amount,
    p.product_margin
HAVING SUM(o.loss_amount) > 0 OR SUM(o.profit) < 0;

SELECT *
FROM vw_loss_alerts
ORDER BY loss_amount DESC, avg_discount DESC
LIMIT 50;


-- Detail view for Power BI drill-through pages

CREATE OR REPLACE VIEW vw_profit_leakage_detail AS
SELECT
    row_id,
    order_id,
    order_date,
    ship_date,
    order_year,
    order_month_num,
    order_month,
    region,
    country,
    state,
    city,
    market,
    market_group,
    segment,
    order_priority,
    ship_mode,
    category,
    sub_category,
    product_id,
    product_name,
    customer_id,
    customer_name,
    sales,
    profit,
    loss_amount,
    discount,
    discount_bucket,
    shipping_cost,
    shipping_cost_pct_of_sales,
    delivery_days,
    delivery_bucket,
    profit_margin,
    is_loss_order,
    CASE
        WHEN is_loss_order = 1 AND discount >= 0.30 THEN 'High discount loss'
        WHEN is_loss_order = 1 AND shipping_cost_pct_of_sales > 0.15 THEN 'Shipping-heavy loss'
        WHEN is_loss_order = 1 THEN 'General loss'
        WHEN profit_margin < 0.05 THEN 'Low margin'
        ELSE 'Healthy'
    END AS risk_flag
FROM vw_orders_clean;


-- Performance note:
-- On a bigger table, an index on (order_date, category, sub_category)
-- would help the trend and Pareto sections. For this dataset size, this runs fine as-is.

-- End of analysis.
-- Main takeaways: losses are concentrated in a few sub-categories, led by Tables.
-- Heavy discounting is the clearest driver, especially once discounts move past 30%.
-- Best next step: connect the views above to Power BI and turn them into a short action-focused dashboard.
