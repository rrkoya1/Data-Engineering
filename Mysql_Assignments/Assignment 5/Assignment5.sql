
-- ASSIGNMENT 5: ADVANCED SQL STRUCTURES: CTE, TEMPORARY TABLE, VIEW   by Ritesh Reddy Koya


USE sakila;

-- 1)VIEWS

/* ------------------------------------------------------------
   VIEW 1: vw_customer_film_master
   - Master list of: customer -> rentals -> inventory -> film
   ------------------------------------------------------------ */
CREATE OR REPLACE VIEW sakila.vw_customer_film_master AS
SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    r.rental_id,
    r.rental_date,
    i.inventory_id,
    f.film_id,
    f.title
FROM sakila.customer c
JOIN sakila.rental r     ON c.customer_id = r.customer_id
JOIN sakila.inventory i  ON r.inventory_id = i.inventory_id
JOIN sakila.film f       ON i.film_id = f.film_id;


/* ------------------------------------------------------------
   VIEW 2: vw_film_category_master
   - Master list of films and their categories (include films with no category)
   - film -> film_category -> category
   ------------------------------------------------------------ */
CREATE OR REPLACE VIEW sakila.vw_film_category_master AS
SELECT
    f.film_id,
    f.title,
    c.category_id,
    c.name AS category
FROM sakila.film f
LEFT JOIN sakila.film_category fc ON f.film_id = fc.film_id
LEFT JOIN sakila.category c       ON fc.category_id = c.category_id;


/* ------------------------------------------------------------
   VIEW 3: vw_org_emails
   - Organization email directory: customers + staff
   ------------------------------------------------------------ */
CREATE OR REPLACE VIEW sakila.vw_org_emails AS
SELECT email, 'customer' AS email_source
FROM sakila.customer
UNION
SELECT email, 'staff' AS email_source
FROM sakila.staff;


-- 2)TEMPORARY TABLES

/* ------------------------------------------------------------
   TEMP TABLE 1: tmp_global_metrics
   - Cache expensive global averages once for the session.
   -(avg payment, avg rental_rate, avg rental_duration).
   ------------------------------------------------------------ */
DROP TEMPORARY TABLE IF EXISTS sakila.tmp_global_metrics;

CREATE TEMPORARY TABLE sakila.tmp_global_metrics AS
SELECT
    (SELECT AVG(rental_rate)      FROM sakila.film)    AS avg_rental_rate,
    (SELECT AVG(rental_duration)  FROM sakila.film)    AS avg_rental_duration,
    (SELECT AVG(amount)           FROM sakila.payment) AS avg_payment_amount;


/* ------------------------------------------------------------
   TEMP TABLE 2: tmp_customer_activity
   - Cache per-customer counts (payments + rentals) for reuse.
   ------------------------------------------------------------ */
DROP TEMPORARY TABLE IF EXISTS sakila.tmp_customer_activity;

CREATE TEMPORARY TABLE sakila.tmp_customer_activity AS
SELECT
    c.customer_id,
    COUNT(DISTINCT p.payment_id) AS payment_count,
    COUNT(DISTINCT r.rental_id)  AS rental_count
FROM sakila.customer c
LEFT JOIN sakila.payment p ON c.customer_id = p.customer_id
LEFT JOIN sakila.rental  r ON c.customer_id = r.customer_id
GROUP BY c.customer_id;


/* ------------------------------------------------------------
   TEMP TABLE 3: tmp_rented_films
   - Cache rented film_ids once.
   - Used for “never rented” logic.
   ------------------------------------------------------------ */
DROP TEMPORARY TABLE IF EXISTS sakila.tmp_rented_films;

CREATE TEMPORARY TABLE sakila.tmp_rented_films AS
SELECT DISTINCT i.film_id
FROM sakila.inventory i
JOIN sakila.rental r ON r.inventory_id = i.inventory_id;



-- 3)CTE

-- A3-Q1) Display all customer details who have made more than 5 payments.
-- Uses: TEMP TABLE (tmp_customer_activity)
/*
Answer:
- tmp_customer_activity already has payment_count per customer.
- CTE filters customer_id with payment_count > 5.
- Join to customer for full details.
*/
WITH active_payers AS (
    SELECT customer_id
    FROM sakila.tmp_customer_activity
    WHERE payment_count > 5
)
SELECT c.*
FROM sakila.customer c
JOIN active_payers ap ON c.customer_id = ap.customer_id;


-- A3-Q2) Find the names of actors who have acted in more than 10 films.

/*
Answer:
- Count films per actor in film_actor.
- Filter count > 10.
- Join to actor to display names.
*/
WITH actor_stats AS (
    SELECT actor_id, COUNT(film_id) AS film_count
    FROM sakila.film_actor
    GROUP BY actor_id
)
SELECT a.actor_id, a.first_name, a.last_name, s.film_count
FROM sakila.actor a
JOIN actor_stats s ON a.actor_id = s.actor_id
WHERE s.film_count > 10;


-- A3-Q3) Find the names of customers who never made a payment.
/*
Answer:
- tmp_customer_activity has payment_count.
- payment_count = 0 means never paid.
*/
WITH never_paid AS (
    SELECT customer_id
    FROM sakila.tmp_customer_activity
    WHERE payment_count = 0
)
SELECT c.customer_id, c.first_name, c.last_name
FROM sakila.customer c
JOIN never_paid np ON c.customer_id = np.customer_id;


-- A3-Q4) Films whose rental rate is higher than the average rental rate of all films.

/*
Answer:
- tmp_global_metrics stores avg_rental_rate once.
- CTE reads it, then filters films above that average.
*/
WITH metrics AS (
    SELECT avg_rental_rate
    FROM sakila.tmp_global_metrics
)
SELECT f.film_id, f.title, f.rental_rate
FROM sakila.film f
JOIN metrics m
WHERE f.rental_rate > m.avg_rental_rate
ORDER BY f.rental_rate DESC;

-- A3-Q5) Titles of films that were never rented.

/*
Answer:
- tmp_rented_films contains film_id values that WERE rented.
- CTE finds films not present in that list.
*/
WITH never_rented AS (
    SELECT f.film_id, f.title
    FROM sakila.film f
    LEFT JOIN sakila.tmp_rented_films rf ON f.film_id = rf.film_id
    WHERE rf.film_id IS NULL
)
SELECT film_id, title
FROM never_rented
ORDER BY title;


-- A3-Q6) Customers who rented in the same month as customer ID 5.

/*
Answer:
- CTE1 finds (year, month) combinations for customer 5.
- CTE2 finds customers renting in those same (year, month).
- Final select returns customer names.
*/
WITH customer5_months AS (
    SELECT DISTINCT YEAR(rental_date) AS yr, MONTH(rental_date) AS mon
    FROM sakila.rental
    WHERE customer_id = 5
),
cohort_customers AS (
    SELECT DISTINCT r.customer_id
    FROM sakila.rental r
    JOIN customer5_months m
      ON YEAR(r.rental_date) = m.yr AND MONTH(r.rental_date) = m.mon
)
SELECT c.customer_id, c.first_name, c.last_name
FROM sakila.customer c
JOIN cohort_customers cc ON c.customer_id = cc.customer_id
ORDER BY c.customer_id;


-- A3-Q7) Staff who handled a payment greater than the average payment amount.

/*
Answer:
- tmp_global_metrics stores avg_payment_amount.
- CTE finds staff_id with any payment > average.
- Join to staff to display names.
*/
WITH avg_pay AS (
    SELECT avg_payment_amount
    FROM sakila.tmp_global_metrics
),
staff_ids AS (
    SELECT DISTINCT p.staff_id
    FROM sakila.payment p
    JOIN avg_pay a
    WHERE p.amount > a.avg_payment_amount
)
SELECT s.staff_id, s.first_name, s.last_name
FROM sakila.staff s
JOIN staff_ids si ON s.staff_id = si.staff_id;


-- A3-Q8) Films whose rental duration is greater than the average rental duration.

/*
Answer:
- tmp_global_metrics stores avg_rental_duration.
- Filter films above it.
*/
WITH avg_dur AS (
    SELECT avg_rental_duration
    FROM sakila.tmp_global_metrics
)
SELECT f.film_id, f.title, f.rental_duration
FROM sakila.film f
JOIN avg_dur d
WHERE f.rental_duration > d.avg_rental_duration
ORDER BY f.rental_duration DESC, f.title;


-- A3-Q9) Customers who have the same address as customer ID 1.

/*
Answer:
- CTE gets address_id of customer 1.
- Return all customers with that same address_id.
*/
WITH cust1 AS (
    SELECT address_id
    FROM sakila.customer
    WHERE customer_id = 1
)
SELECT c.customer_id, c.first_name, c.last_name, c.address_id
FROM sakila.customer c
JOIN cust1 a
WHERE c.address_id = a.address_id
ORDER BY c.customer_id;


-- A3-Q10) Payments greater than the average of all payments.
-- Uses: TEMP TABLE (tmp_global_metrics)
/*
Answer:
- Use cached avg_payment_amount.
- Filter payment rows above it.
*/
WITH avg_pay AS (
    SELECT avg_payment_amount
    FROM sakila.tmp_global_metrics
)
SELECT p.payment_id, p.customer_id, p.staff_id, p.amount, p.payment_date
FROM sakila.payment p
JOIN avg_pay a
WHERE p.amount > a.avg_payment_amount
ORDER BY p.amount DESC, p.payment_date DESC;



-- A4-Q1) List all customers along with the films they have rented.
-- Uses: VIEW (vw_customer_film_master)

/*
Answer:
- VIEW already contains the full join path.
- CTE reads from the view, then selects columns for display.
*/
WITH master AS (
    SELECT customer_id, first_name, last_name, film_id, title, rental_date
    FROM sakila.vw_customer_film_master
)
SELECT *
FROM master
ORDER BY customer_id, rental_date;


-- A4-Q2) All customers + rental count (include customers with zero rentals).

/*
Answer:
- tmp_customer_activity already has rental_count for every customer.
- Join to customer to show names + rental_count.
*/
WITH rental_counts AS (
    SELECT customer_id, rental_count
    FROM sakila.tmp_customer_activity
)
SELECT c.customer_id, c.first_name, c.last_name, rc.rental_count
FROM sakila.customer c
JOIN rental_counts rc ON c.customer_id = rc.customer_id
ORDER BY rc.rental_count DESC, c.customer_id;


-- A4-Q3) Films with their category (include uncategorized films).
-- Uses: VIEW (vw_film_category_master)
/*
Answer:
- VIEW already handles LEFT JOIN logic to include films without category.
- CTE simply reads and outputs.
*/
WITH film_cat AS (
    SELECT film_id, title, category
    FROM sakila.vw_film_category_master
)
SELECT *
FROM film_cat
ORDER BY title;


-- A4-Q4) All customer and staff emails from both tables.
-- Uses: VIEW (vw_org_emails)
/*
Answer:
- This interpretation means “combine email lists”.
- View is built using UNION (similar to FULL OUTER join idea for lists).
*/
WITH emails AS (
    SELECT email, email_source
    FROM sakila.vw_org_emails
)
SELECT *
FROM emails
ORDER BY email_source, email;


-- A4-Q5) Actors who acted in the film "ACADEMY DINOSAUR".

/*
Answer:
- CTE1 finds the film_id for the title.
- CTE2 finds actor_id values linked to that film_id.
- Final select returns actor names.
*/
WITH target_film AS (
    SELECT film_id
    FROM sakila.film
    WHERE title = 'ACADEMY DINOSAUR'
),
actor_ids AS (
    SELECT fa.actor_id
    FROM sakila.film_actor fa
    JOIN target_film tf ON fa.film_id = tf.film_id
)
SELECT a.actor_id, a.first_name, a.last_name
FROM sakila.actor a
JOIN actor_ids ai ON a.actor_id = ai.actor_id
ORDER BY a.last_name, a.first_name;


-- A4-Q6) Stores and total number of staff in each store (include stores with 0 staff).

/*
Answer:
- CTE counts staff per store_id.
- LEFT JOIN ensures all stores appear.
*/
WITH staff_counts AS (
    SELECT store_id, COUNT(staff_id) AS staff_count
    FROM sakila.staff
    GROUP BY store_id
)
SELECT st.store_id, COALESCE(sc.staff_count, 0) AS staff_count
FROM sakila.store st
LEFT JOIN staff_counts sc ON st.store_id = sc.store_id
ORDER BY st.store_id;


-- A4-Q7) Customers who have rented films more than 5 times (name + rental count).

/*
Answer:
- tmp_customer_activity contains rental_count.
- Filter rental_count > 5, then show customer names.
*/
WITH heavy_renters AS (
    SELECT customer_id, rental_count
    FROM sakila.tmp_customer_activity
    WHERE rental_count > 5
)
SELECT c.customer_id, c.first_name, c.last_name, hr.rental_count
FROM sakila.customer c
JOIN heavy_renters hr ON c.customer_id = hr.customer_id
ORDER BY hr.rental_count DESC, c.customer_id;

