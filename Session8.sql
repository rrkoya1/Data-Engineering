
-- Session 8 Notes — CTE, Temporary Tables, Views


USE sakila;

-- 1) COMMON TABLE EXPRESSIONS (CTE)

/* 
   What is a CTE?
   - A CTE is a temporary named result set.
   - It improves readability and reusability.
   - Defined using: WITH cte_name AS ( ... )

   Key idea:
   - Scope is query-level (only for that query).
   - Once query finishes, CTE disappears.
    */

-- 1.1 Simple CTE Example
-- Find customers and their total payment amount
WITH customer_payments AS (
    SELECT customer_id, SUM(amount) AS total_spent
    FROM sakila.payment
    GROUP BY customer_id
)
SELECT customer_id, total_spent
FROM customer_payments
ORDER BY total_spent DESC;

-- Explanation:
-- - CTE runs first and creates a temporary output table: customer_payments
-- - Then main SELECT reads from it like a normal table.

-- 1.2 CTE with Join (Readable)
-- Show customer names + total amount they spent

WITH customer_payments AS (
    SELECT customer_id, SUM(amount) AS total_spent
    FROM sakila.payment
    GROUP BY customer_id
)
SELECT c.customer_id, c.first_name, c.last_name, cp.total_spent
FROM sakila.customer c
JOIN customer_payments cp ON c.customer_id = cp.customer_id
ORDER BY cp.total_spent DESC;

-- Explanation:
-- - We avoid repeating SUM(amount) query again and again
-- - CTE makes it cleaner


-- 1.3 Multiple CTEs in one query
-- We use "," to define more than one CTE

WITH
customer_payments AS (
    SELECT customer_id, SUM(amount) AS total_spent
    FROM sakila.payment
    GROUP BY customer_id
),
top_customers AS (
    SELECT customer_id, total_spent
    FROM customer_payments
    ORDER BY total_spent DESC
    LIMIT 5
)
SELECT tc.customer_id, c.first_name, c.last_name, tc.total_spent
FROM top_customers tc
JOIN sakila.customer c ON tc.customer_id = c.customer_id;

-- Explanation:
-- - First CTE makes totals
-- - Second CTE selects top 5 customers
-- - Final query shows names


-- 1.4 Validation Query idea
-- Always validate your CTE results by checking small parts

WITH customer_payments AS (
    SELECT customer_id, COUNT(*) AS total_payments
    FROM sakila.payment
    GROUP BY customer_id
)
SELECT *
FROM customer_payments
WHERE total_payments > 5;

-- Explanation:
-- - This kind of check is called a "validation query"
-- - Helps confirm your logic works

/* 
    CTE vs Subquery (Readability / Logic)
   
   - Subqueries are "inside-out":
       You must read the middle query first to understand the outer query.
       (This becomes confusing when there are 2+ nested levels.)

   - CTEs are "top-to-bottom":
       You define the result first (WITH ...), then use it like a table.
       This makes the query easier to follow like a story.

   Note:
   Use CTEs when you have more than 2 levels of nesting,
   otherwise queries become hard to debug.
    */



-- 2) RECURSIVE CTE


/* 
   What is a Recursive CTE?
   - A CTE that calls itself to generate a sequence.

   Parts:
   1) Anchor member    -> starting row (base case)
   2) Recursive member -> repeats until condition stops

   Syntax:
   WITH RECURSIVE cte AS (
       anchor_query
       UNION ALL
       recursive_query referencing cte
   )
   SELECT * FROM cte;
    */

/* 
   Recursive CTE Terminator Warning
   
   Recursive CTE has two parts:
   1) Anchor member  -> starting row
   2) Recursive member -> repeats step-by-step

   VERY IMPORTANT:
   Always include a stopping condition (terminator), such as:
       WHERE n < 10

   If you forget the WHERE condition:
   - recursion becomes infinite
   - query may crash or freeze the DB server
    */


-- 2.1 Recursive CTE Example: Generate numbers 1 to 10

WITH RECURSIVE numbers AS (
    SELECT 1 AS n         -- Anchor member
    UNION ALL
    SELECT n + 1          -- Recursive member
    FROM numbers
    WHERE n < 10
)
SELECT * FROM numbers;

-- Explanation:
-- - Starts with 1
-- - keeps adding +1 until n reaches 10


-- 2.2 Recursive CTE Example: Last 10 rental days

WITH RECURSIVE dates AS (
    SELECT DATE(MAX(rental_date)) - INTERVAL 9 DAY AS rental_day
    FROM sakila.rental

    UNION ALL

    SELECT rental_day + INTERVAL 1 DAY
    FROM dates
    WHERE rental_day + INTERVAL 1 DAY <= (
        SELECT DATE(MAX(rental_date))
        FROM sakila.rental
    )
)
SELECT d.rental_day, COUNT(r.rental_id) AS rentals
FROM dates d
LEFT JOIN sakila.rental r ON DATE(r.rental_date) = d.rental_day
GROUP BY d.rental_day;

-- Explanation:
-- - Anchor member: max rental date - 9 days
-- - Recursive member: adds 1 day until max rental date
-- - Then LEFT JOIN counts rentals per day
-- - Shows 0 rental days also (because LEFT JOIN)

/*
 Validation query (manual check):
 
 select date(rental_date), count(*)
 from sakila.rental
 group by date(rental_date);
*/


-- 3) TEMPORARY TABLES


/* 
   Temporary Table:
   - Created for the current session only.
   - Stored in memory (or temporary space).
   - Exists until:
       * session ends OR
       * you DROP it manually

   Why use it?
   - Like saving intermediate results
   - Avoid writing long queries repeatedly
   - Useful after joins when analysis is bigger
    */


-- 3.1 Create Temporary Table for Top 5 Categories

DROP TEMPORARY TABLE IF EXISTS sakila.top_categories;

CREATE TEMPORARY TABLE sakila.top_categories AS
SELECT c.name AS category_name, c.category_id, COUNT(*) AS rental_count
FROM sakila.rental r
JOIN sakila.inventory i ON r.inventory_id = i.inventory_id
JOIN sakila.film f ON i.film_id = f.film_id
JOIN sakila.film_category fc ON f.film_id = fc.film_id
JOIN sakila.category c ON fc.category_id = c.category_id
GROUP BY c.name, c.category_id
ORDER BY rental_count DESC
LIMIT 5;

-- Use the temporary table
SELECT * FROM sakila.top_categories;

-- Explanation:
-- - Temporary table stores the results
-- - Now we can reuse without running heavy join again



-- 3.2 Reuse Temporary Table for other queries

SELECT tc.category_name, f.title
FROM sakila.top_categories tc
JOIN sakila.film_category fc ON tc.category_id = fc.category_id
JOIN sakila.film f ON fc.film_id = f.film_id
ORDER BY tc.category_name, f.title;

-- Explanation:
-- - This proves temp table is reusable in session
-- - Great for analysis workflows


-- 4) VIEWS


/* 
   View:
   - Virtual table
   - Does NOT store data
   - It stores the SQL query
   - Each time you select from the view, the query runs

   Benefits:
   - Clean + reusable
   - Security (hide columns)
   - Abstraction (users query view without knowing joins)

   Syntax:
   CREATE OR REPLACE VIEW view_name AS SELECT ...
    */


-- 4.1 Example View: Customers most recent rental date

DROP VIEW IF EXISTS sakila.recent_rentals;

CREATE OR REPLACE VIEW sakila.recent_rentals AS
SELECT r.customer_id AS cstd_id, MAX(r.rental_date) AS ruchik
FROM sakila.rental r
GROUP BY r.customer_id;

-- Use the view
SELECT * FROM sakila.recent_rentals;

-- Explanation:
-- - View stores query only
-- - Looks like a table, but it’s virtual



-- 4.2 Use view with customer table

SELECT c.first_name, c.last_name, rr.ruchik
FROM sakila.customer c
JOIN sakila.recent_rentals rr ON c.customer_id = rr.cstd_id;

-- Explanation:
-- - View acts as clean reusable query layer

/* 
    VIEW Performance Warning
   
   Important:
   A VIEW does NOT store data.
   It is a "virtual table" (saved query).

   Every time you query a view:
   -> MySQL re-runs the entire JOIN / GROUP BY logic again.

   If underlying tables have millions of rows:
   - Views can become slow
   - Temporary Tables are faster for repeated use
   - Other DBs support "Materialized Views" (stores computed data)
    */

