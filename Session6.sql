
-- SESSION 6 also session3 part3
-- Database used: sakila

-- 1) STRING FUNCTIONS (Review)

/* ------------------------------------------------------------
   1.1 SUBSTRING()
   Extract part of a string.
   SUBSTRING(str, start, length)
   start begins at 1 (not 0)
   ------------------------------------------------------------ */
SELECT title, SUBSTRING(title, 1, 3) AS short_title 
FROM sakila.film;


/* ------------------------------------------------------------
   1.2 CONCAT()
   Joins strings together.
   Useful for formatting names/emails/display fields
   ------------------------------------------------------------ */
SELECT CONCAT(first_name, '@ ', last_name) AS full_name 
FROM sakila.customer;


/* ------------------------------------------------------------
   1.3 LENGTH()
   Returns number of bytes in a string (MySQL)
   Used to filter by string length
   ------------------------------------------------------------ */
SELECT title, LENGTH(title) AS title_length 
FROM sakila.film 
WHERE LENGTH(title) > 15;

-- 2) SUBSTRING WITH LOCATE + SUBSTRING_INDEX

/* ------------------------------------------------------------
   LOCATE(substring, string)
   Gives position of substring inside string.
   We use it here to find '@' position inside email.
   ------------------------------------------------------------ */

select email from sakila.customer;

SELECT email,
       SUBSTRING(email, LOCATE('@', email)+1) AS domain
FROM sakila.customer;


/* ------------------------------------------------------------
   substring_index(str, delimiter, count)
   If count is positive -> returns from LEFT side
   If count is negative -> returns from RIGHT side

   Here:
   1) take everything after '@'
   2) then take only first part before '.'
   ------------------------------------------------------------ */
SELECT 
  email,
  substring_index(SUBSTRING(email, LOCATE('@', email) + 1), '.', 1) AS domain
FROM 
  sakila.customer;


/* ------------------------------------------------------------
   count = -1 means: take last part AFTER '@'
   Example: "a@gmail.com" -> "gmail.com"
   ------------------------------------------------------------ */
select substring_index(email,'@', -1) from sakila.customer;

-- 3) FILTERING USING UPPER() / LOWER()

/* ------------------------------------------------------------
   UPPER() makes text uppercase so search becomes safer.
   LIKE is not always case sensitive depending on collation,
   but using UPPER() makes it consistent.
   ------------------------------------------------------------ */
SELECT title
FROM sakila.film
WHERE UPPER(title) LIKE '%LOVELY%' OR UPPER(title) LIKE '%MAN';


/* Lowercase output for display */
select title, lower(title) as lower_titles
FROM sakila.film;

-- 4) GROUP BY + ORDER BY (Counting groups)

/* ------------------------------------------------------------
   LEFT(title, 1) -> first character
   RIGHT(title, 1) -> last character
   Group films by first+last letter, then count them
   ------------------------------------------------------------ */
SELECT LEFT(title, 1) AS first_letter, right(title,1) as last_letter, COUNT(*) AS film_count
FROM sakila.film
GROUP BY LEFT(title, 1), right(title,1) 
ORDER BY film_count DESC;


SELECT LEFT(title,1) AS first_letter, right(title, 1) as last_letter, title 
from sakila.film;

-- 5) CASE STATEMENT (CONTROL FLOW)

/* ------------------------------------------------------------
   CASE is SQL control flow.
   Used like IF-ELSE logic.
   Here we group customers based on first letter of last_name.
   ------------------------------------------------------------ */
SELECT last_name,
       CASE 
           WHEN LEFT(last_name, 1) BETWEEN 'A' AND 'M' THEN 'Group A-M'
           WHEN LEFT(last_name, 1) BETWEEN 'N' AND 'Z' THEN 'Group N-Z'
           ELSE 'Other'
       END AS group_label
FROM sakila.customer;

-- 6) REPLACE()

/* ------------------------------------------------------------
   REPLACE(str, old, new)
   Replaces ALL occurrences
   ------------------------------------------------------------ */
SELECT title, REPLACE(title, 'A', 'x') AS cleaned_title
FROM sakila.film
WHERE title LIKE '% ' '%';

-- 7) REGEXP (Pattern matching)

/* ------------------------------------------------------------
   REGEXP allows advanced matching patterns.
   Pattern: '[^aeiouAEIOU]{3}'

   Meaning:
   [aeiouAEIOU]   -> vowels set
   [^...]         -> NOT vowels (consonants or other chars)
   {3}            -> exactly 3 in a row

   So: find customers where last_name contains 3 consecutive NON-vowels
   ------------------------------------------------------------ */
SELECT customer_id, last_name
FROM sakila.customer
WHERE last_name REGEXP '[^aeiouAEIOU]{3}'; -- decode


/* ------------------------------------------------------------
   Pattern: '[aeiouAEIOU]$'

   $ means end of string.
   So: titles that end with a vowel.
   ------------------------------------------------------------ */
SELECT title
FROM sakila.film
WHERE title REGEXP '[aeiouAEIOU]$';


/* ------------------------------------------------------------
   Count how many film titles end with each vowel.
   right(title,1) gives last letter of title
   ------------------------------------------------------------ */
select right(title,1), count(*)
FROM sakila.film
WHERE title REGEXP '[aeiouAEIOU]$'
group by right(title,1)
;

-- 8) MATH + AGGREGATION

/* ------------------------------------------------------------
   rental_rate * 2 is numeric multiplication
   This is just computed output (no data change)
   ------------------------------------------------------------ */
SELECT title, rental_rate, rental_rate * 2 AS double_rate
FROM sakila.film;


/* ------------------------------------------------------------
   Aggregations:
   COUNT() counts rows
   SUM() adds values
   avg_payment = SUM / COUNT
   grouped by customer_id
   ------------------------------------------------------------ */
SELECT customer_id,
       COUNT(payment_id) AS payments,
       SUM(amount) AS total_paid,
       SUM(amount) / COUNT(payment_id) AS avg_payment
FROM sakila.payment
GROUP BY customer_id;

-- 9) ALTER TABLE + UPDATE (THIS MODIFIES DATABASE)


/* ------------------------------------------------------------
   Adds a new column to sakila.film.
   This modifies schema (DDL)
   ------------------------------------------------------------ */
ALTER TABLE sakila.film
ADD COLUMN cost_efficiency DECIMAL(6,2);


/* ------------------------------------------------------------
   UPDATE modifies table contents (DML)
   We compute cost_efficiency = replacement_cost / length
   Only rows where length is not NULL get updated.
   ------------------------------------------------------------ */
UPDATE sakila.film
SET cost_efficiency = replacement_cost / length
WHERE length IS NOT NULL;


/* View result */
select * from sakila.film;


-- 10) DATES (DATEDIFF, MONTH, NOW)

/* ------------------------------------------------------------
   DATEDIFF(date1, date2)
   Returns difference in days: date1 - date2
   Used here to calculate how many days rented
   ------------------------------------------------------------ */
SELECT rental_id, DATEDIFF(return_date, rental_date) AS days_rented
FROM sakila.rental
WHERE return_date IS NOT NULL;


/* Get numeric month from a datetime */
select month(last_update) from sakila.film;


SELECT payment_date FROM sakila.payment;


/* ------------------------------------------------------------
   DATE(datetime) extracts only date part.
   Group payments per day.
   ------------------------------------------------------------ */
SELECT DATE(payment_date) AS pay_date, SUM(amount) AS total_paid
FROM sakila.payment
GROUP BY DATE(payment_date)
ORDER BY pay_date DESC;


/* ------------------------------------------------------------
   Payments in last 24 hours
   NOW() gives current datetime
   ------------------------------------------------------------ */
select * from sakila.payment;

SELECT customer_id, amount, payment_date
FROM sakila.payment
WHERE payment_date >= NOW() - INTERVAL 1 DAY;


/* Latest payment date */
select max(payment_date) FROM sakila.payment;


/* Payments in last 1 day relative to latest payment */
SELECT customer_id, amount, payment_date
FROM sakila.payment
WHERE payment_date >= (
    SELECT MAX(payment_date) - INTERVAL 1 day
    FROM sakila.payment
);


select now()  - INTERVAL 1 DAY as yesterday;


SELECT CONCAT('Today is: ', CURDATE()) AS message;
SELECT CONCAT('Today is: ', now()) AS message;

SELECT NOW(), CURDATE(), CURRENT_TIME;


-- 11) SUBQUERIES

/* ------------------------------------------------------------
   A subquery is a query inside another query.

   3 common uses:
   1) subquery inside WHERE
   2) subquery inside SELECT
   3) derived tables (subquery in FROM)

   Weakness:
   - sometimes subquery returns multiple rows and fails
   - not reusable
   - performance can be slower than JOINs
   ------------------------------------------------------------ */


-- ------------------------------------------------------------
-- 11.1 Subquery in WHERE (IN)
-- ------------------------------------------------------------
SELECT first_name, last_name
FROM sakila.customer
WHERE address_id IN (
    SELECT address_id
    FROM sakila.customer
     #WHERE customer_id = 1
);


-- ------------------------------------------------------------
-- 11.2 Subquery with GROUP BY + HAVING
-- Find actors who appeared in > 10 films
-- ------------------------------------------------------------
SELECT actor_id, first_name, last_name
FROM sakila.actor
WHERE actor_id IN (
    SELECT actor_id
    FROM sakila.film_actor
    GROUP BY actor_id
    HAVING COUNT(film_id) > 10
);


-- ------------------------------------------------------------
-- 11.3 Subquery in SELECT (correlated)
-- Runs for each actor row
-- ------------------------------------------------------------
SELECT actor_id,
       first_name,
       last_name,
       (
           SELECT COUNT(*)
           FROM sakila.film_actor
           WHERE film_actor.actor_id = actor.actor_id
       ) AS film_count
FROM sakila.actor;


-- 12) DERIVED TABLES

/* ------------------------------------------------------------
   Derived table = temporary virtual table result
   You can join it like a normal table.
   ------------------------------------------------------------ */

SELECT a.actor_id, a.first_name, a.last_name, fa.film_count
FROM sakila.actor a
JOIN (
    SELECT actor_id, COUNT(film_id) AS film_count
    FROM sakila.film_actor
    GROUP BY actor_id
    HAVING COUNT(film_id) > 10
) fa ON a.actor_id = fa.actor_id;


SELECT customer_id, total_spent
FROM (
    SELECT customer_id, SUM(amount) AS total_spent
    FROM sakila.payment
    GROUP BY customer_id
    ORDER BY total_spent DESC
    LIMIT 5
) AS top_customers;


SELECT *
FROM (
    SELECT last_name,
           CASE 
               WHEN LEFT(last_name, 1) BETWEEN 'A' AND 'M' THEN 'Group A-M'
               WHEN LEFT(last_name, 1) BETWEEN 'N' AND 'Z' THEN 'Group N-Z'
               ELSE 'Other'
           END AS group_label
    FROM sakila.customer
) AS grouped_customers 
WHERE group_label = 'Group N-Z';


/* Execution order:
   FROM -> WHERE -> SELECT
*/
 -- FROM ---- > Where --->  select 

-- 13) SUBQUERY WHEN TO USE + FAILURE EXAMPLE

/* ------------------------------------------------------------
   Compare values against aggregate result (AVG)
   ------------------------------------------------------------ */
SELECT customer_id, amount
FROM sakila.payment
WHERE amount > (
    SELECT AVG(amount)
    FROM sakila.payment
);


/* ------------------------------------------------------------
   Example where subquery FAILS:
   This subquery returns many rows (not 1),
   so it breaks because SELECT needs single value.
   ------------------------------------------------------------ */
SELECT first_name,
       (SELECT address_id FROM sakila.address WHERE district = 'California'  ) AS cali_address
FROM sakila.customer;


-- 14) CORRELATED SUBQUERIES

/* ------------------------------------------------------------
   Correlated subquery depends on outer row.
   Here:
   for each film f, count how many actors appear in it.
   ------------------------------------------------------------ */
SELECT title,
  (SELECT COUNT(*)
   FROM sakila.film_actor fa
   WHERE fa.film_id = f.film_id) AS actor_count
FROM sakila.film f;


/* ------------------------------------------------------------
   For each payment row p1, compare amount against
   avg payment amount of same customer (p2)
   ------------------------------------------------------------ */
SELECT payment_id, customer_id, amount, payment_date
FROM sakila.payment p1
WHERE amount > (
    SELECT AVG(amount)
    FROM sakila.payment p2
    WHERE p2.customer_id = p1.customer_id
);

select amount from sakila.customer;

-- 15) JOINS (Basic Introduction)

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
-- 15.1 INNER JOIN
-- Shows film and its language
-- --------------------------
SELECT f.title, l.name AS language
FROM sakila.film f
INNER JOIN sakila.language l ON f.language_id = l.language_id;

-- SELECT f.title, l.name AS language
-- FROM sakila.film f
-- INNER JOIN sakila.language l ON f.language_id = l.language_id;



-- --------------------------
-- 15.2 LEFT JOIN
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

-- 16) FULL OUTER JOIN (MySQL workaround)

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


-- 17) SELF JOIN

/* ------------------------------------------------------------
   Self join = join table with itself.
   Here we join staff with staff to find pairs in same store.
   ------------------------------------------------------------ */

SELECT s1.staff_id, s2.staff_id, s1.store_id
FROM sakila.staff s1
JOIN sakila.staff s2 ON s1.store_id = s2.store_id
WHERE s1.staff_id <> s2.staff_id;


select * from sakila.staff;


-- 18) SELF JOIN DEMO TABLE

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


-- 19) WHERE EXISTS

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

-- 20) UNION EMAIL EXAMPLE

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

/*
In database design, tables can be connected to each other using relationships, usually through Primary Keys (PK) and Foreign Keys (FK).
 In the images from class, we discussed 4 common relationship types: 1:1, 1:Many, Many:1, and Many:Many, and we also connected them with how INNER JOIN and LEFT JOIN behave.

A 1 : 1 (One-to-One) relationship means each record in Table A matches exactly one record in Table B, and vice-versa.
 In your example, this is shown as users ↔ user_profile. Every user has only one profile, and every profile belongs to only one user. 
 Usually, the profile table contains a column like user_id which is both a foreign key referencing users and also unique, so no two profiles can point to the same user.

A 1 : Many (One-to-Many) relationship means one record in the parent table can be linked to many records in the child table.
 In the images, this is shown as users → orders. One user can create multiple orders, but each order belongs to only one user. 
 This happens by storing user_id as an FK inside the orders table. This relationship is one of the most common in real databases.

A Many : 1 (Many-to-One) relationship is basically the same as 1:Many, but described from the opposite side. 
So instead of saying “one user has many orders,” we say: many orders belong to one user.
 The structure is the same — multiple rows in orders point to the same user in users.

A Many : Many (Many-to-Many) relationship means many records in Table A can connect to many records in Table B.
 In your example, this is shown using the friendships table (like friendships(user_id, friend_id)). 
 Here, a user can have many friends, and each friend can also be connected to many users. 
 Since SQL databases cannot store many-to-many directly in one FK column, we always solve it using a bridge/junction table (also called mapping table), 
 which holds the IDs from both sides.

Finally, we connected this concept to JOIN behavior:

INNER JOIN returns only rows that match in both tables (common data only).

LEFT JOIN returns all rows from the left table and matches from the right table (if there is no match, the right side becomes NULL).
*/