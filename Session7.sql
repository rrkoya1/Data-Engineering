-- Session 7 continue from session 6 also session3 part 3
-- 1) JOINS 

/* ------------------------------------------------------------
   JOIN connects tables using common keys (PK/FK).
   Useful because subqueries can get slow, harder to reuse.

   Types introduced:
   - INNER JOIN (only matches)
   - LEFT JOIN (keep left rows even if no match)
   - FULL OUTER JOIN (MySQL doesn't support directly)
   - SELF JOIN
   ------------------------------------------------------------ */


-- --------------------------
-- 1.1 INNER JOIN
-- Shows film and its language
-- --------------------------
SELECT f.title, l.name AS language
FROM sakila.film f
INNER JOIN sakila.language l ON f.language_id = l.language_id;

-- SELECT f.title, l.name AS language
-- FROM sakila.film f
-- INNER JOIN sakila.language l ON f.language_id = l.language_id;



-- --------------------------
-- 1.2 LEFT JOIN
-- film -> film_category -> category
-- keeps all films even if category missing
-- --------------------------
SELECT f.title, c.name AS category
FROM sakila.film f
LEFT JOIN sakila.film_category fc ON f.film_id = fc.film_id
LEFT JOIN sakila.category c ON fc.category_id = c.category_id;


/* customers + rentals (customers with no rentals still appear) */
SELECT c.customer_id, c.first_name, r.rental_id
FROM sakila.customer c
LEFT JOIN sakila.rental r ON c.customer_id = r.customer_id;

-- 2) FULL OUTER JOIN (MySQL workaround)

/* ------------------------------------------------------------
   MySQL doesn't have FULL OUTER JOIN directly.
   We simulate it using:
   LEFT JOIN
   UNION
   RIGHT JOIN
   ------------------------------------------------------------ */

-- List all actors and films they acted in (even if unmatched either side)
SELECT a.actor_id, a.first_name, fa.film_id
FROM sakila.actor a
LEFT JOIN sakila.film_actor fa ON a.actor_id = fa.actor_id

UNION

SELECT a.actor_id, a.first_name, fa.film_id
FROM sakila.actor a
RIGHT JOIN sakila.film_actor fa ON a.actor_id = fa.actor_id;


-- List all customers and rentals, including unmatched
SELECT c.customer_id, r.rental_id
FROM sakila.customer c
LEFT JOIN sakila.rental r ON c.customer_id = r.customer_id

UNION

SELECT c.customer_id, r.rental_id
FROM sakila.customer c
RIGHT JOIN sakila.rental r ON c.customer_id = r.customer_id;


-- 3) SELF JOIN

/* ------------------------------------------------------------
   Self join = join table with itself.
   Here we join staff with staff to find pairs in same store.
   ------------------------------------------------------------ */

SELECT s1.staff_id, s2.staff_id, s1.store_id
FROM sakila.staff s1
JOIN sakila.staff s2 ON s1.store_id = s2.store_id
WHERE s1.staff_id <> s2.staff_id;


select * from sakila.staff;


-- 4) SELF JOIN DEMO TABLE

-- 1. Create the staff_demo table
CREATE TABLE sakila.staff_demo (
    staff_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    store_id INT
);

-- 2. Insert sample data
INSERT INTO sakila.staff_demo (staff_id, first_name, store_id) VALUES
(1, 'Alice', 1),
(2, 'Bob', 1),
(3, 'Charlie', 2),
(4, 'Diana', 2),
(5, 'Ethan', 1);

-- 3. Self join: staff pairs working in same store
SELECT 
    s1.staff_id AS staff_1_id,
    s1.first_name AS staff_1_name,
    s2.staff_id AS staff_2_id,
    s2.first_name AS staff_2_name,
    s1.store_id
FROM sakila.staff_demo s1
JOIN sakila.staff_demo s2
  ON s1.store_id = s2.store_id
  AND s1.staff_id <> s2.staff_id
ORDER BY s1.store_id, s1.staff_id;


-- 5) WHERE EXISTS

/* ------------------------------------------------------------
   EXISTS checks if at least one matching row exists.
   It returns TRUE/FALSE.
   Here: check if payment customer also has rentals.
   ------------------------------------------------------------ */
SELECT DISTINCT p.customer_id,p.rental_id
FROM sakila.payment p
WHERE EXISTS (
    SELECT 1
    FROM sakila.rental r
    WHERE r.customer_id = p.customer_id
);

-- 6) UNION EMAIL EXAMPLE

/* ------------------------------------------------------------
   This example is just to show UNION concept.
   Using "ON 1 = 0" means join never matches -> produces NULL
   then UNION brings results together
   ------------------------------------------------------------ */

SELECT s.email FROM sakila.customer c
LEFT JOIN sakila.staff s ON 1 = 0

UNION
SELECT s.email FROM sakila.staff s
LEFT JOIN sakila.customer c ON 1 = 0;