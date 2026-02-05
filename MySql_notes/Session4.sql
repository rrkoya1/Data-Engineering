/* 
   Session 4 — DQL (Data Query Language) + Concepts (MySQL / Sakila)
   What is DQL?
   - DQL is mainly the SELECT part of SQL: querying/reading data (no modification).

    */

USE sakila;

/* 
   1) SELECT (Read data)
   
   SELECT is used to retrieve data from tables.
   - SELECT *    -> all columns
   - SELECT col1 -> specific columns
   */

-- All columns + all rows (full table view)
SELECT * FROM sakila.actor;

-- Only the columns you need 
SELECT actor_id, first_name, last_name
FROM sakila.actor;


/* 
   2) DISTINCT (Remove duplicates)
   
   DISTINCT removes duplicate rows from the selected output.
   Example use:
   - What unique ratings exist?
   - What unique first names exist?
    */

SELECT DISTINCT first_name
FROM sakila.actor;

SELECT DISTINCT rating
FROM sakila.film;


/* 
   3) COUNT 
   
   COUNT is an aggregate function:
   - COUNT(*) counts rows (including rows with NULLs)
   - COUNT(column) counts NON-NULL values in that column
   - COUNT(DISTINCT column) counts unique non-NULL values
    */

-- How many films (rows) exist?
SELECT COUNT(*) AS total_films
FROM sakila.film;

-- Count non-NULL titles (usually same as COUNT(*) if title is NOT NULL)
SELECT COUNT(title) AS title_count
FROM sakila.film;

-- Count unique titles
SELECT COUNT(DISTINCT title) AS distinct_title_count
FROM sakila.film;

-- Compare: count all first_name values vs distinct first names
SELECT COUNT(first_name) AS count_first_names
FROM sakila.actor;

SELECT COUNT(DISTINCT first_name) AS distinct_first_names
FROM sakila.actor;


/* 
   4) Selecting specific columns + LIMIT
   
   LIMIT is used to show only the first N rows.
    */

SELECT first_name, last_name
FROM sakila.actor
LIMIT 5;


/* 
   5) WHERE (Filtering rows)
   
   WHERE filters rows BEFORE they are returned.
   Common operators:
   =, <, >, <=, >=, <>
   AND, OR, NOT
    */

-- Example: find films where original_language_id is missing (NULL)
SELECT *
FROM sakila.film
WHERE original_language_id IS NULL;

-- Example: numeric condition
SELECT *
FROM sakila.film
WHERE length >= 92;

-- Example: multiple conditions with AND
SELECT *
FROM sakila.film
WHERE rating = 'R' AND length >= 92;


/* 
   6) ORDER BY (Sorting)
   
   ORDER BY sorts output:
   - ASC  (ascending) default
   - DESC (descending)
    */

-- View rental_rate values
SELECT rental_rate
FROM sakila.film;

-- Sort by rental_rate high -> low
SELECT rental_rate
FROM sakila.film
ORDER BY rental_rate DESC;

-- Sort by rental_rate low -> high
SELECT rental_rate
FROM sakila.film
ORDER BY rental_rate ASC;


/* 
   7) AND / OR (Combining conditions)
   
   AND: both conditions must be true
   OR : at least one condition must be true
   */

-- AND example
SELECT *
FROM sakila.film
WHERE rating = 'PG' AND rental_duration = 5
ORDER BY rental_rate ASC;

-- OR example
SELECT *
FROM sakila.film
WHERE rating = 'PG' OR rental_duration = 5
ORDER BY rental_rate ASC;


/* 
   8) NOT, <> (Not equal), IN / NOT IN
   
   <> means "not equal"
   NOT IN excludes a list of values
    */

-- Exclude rental_duration values 6,7,3
SELECT *
FROM sakila.film
WHERE rental_duration NOT IN (6, 7, 3)
ORDER BY rental_rate ASC;

-- Not equal to 6 using <>
SELECT *
FROM sakila.film
WHERE rental_duration <> 6
ORDER BY rental_rate ASC;

-- Not equal to 6 using NOT
SELECT *
FROM sakila.film
WHERE NOT rental_duration = 6
ORDER BY rental_rate ASC;


/* 
   9) Parentheses for multiple conditions
   
   Use parentheses to control logic:
   A AND (B OR C) is different from (A AND B) OR C
   */

SELECT *
FROM sakila.film
WHERE rental_duration = 6
  AND (rating = 'G' OR rating = 'PG')
ORDER BY rental_rate ASC;


/* 
   10) LIKE (Pattern matching) with % and _
   
   LIKE is used for text pattern matching:
   - %  : 0 or more characters
   - _  : exactly 1 character

   NOTE about case sensitivity:
   - In MySQL, LIKE can be case-insensitive depending on collation.
   - Some DBs are strict case-sensitive by default.
    */

-- Cities that start with A and end with s (A ... s)
SELECT city
FROM sakila.city
WHERE city LIKE 'A%s';

-- Exactly 3 characters, middle character is 'a' ( _ a _ )
SELECT city
FROM sakila.city
WHERE city LIKE '_a_';


/* 
   11) NULL / NOT NULL (Missing values)
   
   In SQL, NULL means "missing/unknown".
   - You cannot check NULL using = or <>.
   - Use: IS NULL or IS NOT NULL
    */

-- Rentals never returned (return_date is NULL)
SELECT rental_id, inventory_id, customer_id, return_date
FROM sakila.rental
WHERE return_date IS NULL;

-- Rentals that were returned (return_date exists)
SELECT rental_id, inventory_id, customer_id, return_date
FROM sakila.rental
WHERE return_date IS NOT NULL
LIMIT 10;


/* 
   12) BETWEEN (Range filter)
   
   BETWEEN is inclusive:
   return_date BETWEEN 'A' AND 'B' includes both endpoints.
   Good for dates and numbers.
    */

SELECT rental_id, inventory_id, customer_id, return_date
FROM sakila.rental
WHERE return_date BETWEEN '2005-05-26' AND '2005-05-30';


/* 
   13) GROUP BY (Aggregation per group)
   
   GROUP BY groups rows so aggregates can be calculated per group.
   Common aggregates:
   - COUNT(), SUM(), AVG(), MIN(), MAX()

   Rule:
   - If you SELECT a non-aggregated column, it must appear in GROUP BY.
    */

-- Rentals per customer
SELECT customer_id, COUNT(*) AS rental_count
FROM sakila.rental
GROUP BY customer_id
ORDER BY rental_count DESC;

-- Same idea: "occurrences" shows repeated customer_id counts
SELECT customer_id, COUNT(*) AS occurrences
FROM sakila.rental
GROUP BY customer_id
ORDER BY occurrences DESC;


/* 
   14) HAVING (Filter groups after grouping)
   
   WHERE filters rows BEFORE GROUP BY
   HAVING filters groups AFTER GROUP BY
    */

-- Customers who have <= 30 rentals
SELECT customer_id, COUNT(*) AS rental_count
FROM sakila.rental
GROUP BY customer_id
HAVING COUNT(*) <= 30
ORDER BY rental_count DESC;

-- Total payment per customer, keep only customers with total > 100
SELECT customer_id, SUM(amount) AS total_payment
FROM sakila.payment
GROUP BY customer_id
HAVING SUM(amount) > 100
ORDER BY total_payment DESC;


/* 
   15) Aliases (AS)
   
   Aliases make output and queries more readable.
   - Column alias: SUM(amount) AS total_payment
   - Table alias : sakila.actor AS a
    */

-- Column alias
SELECT customer_id, SUM(amount) AS total_payment
FROM sakila.payment
GROUP BY customer_id
ORDER BY total_payment DESC;

-- Table alias
SELECT a.first_name, a.last_name
FROM sakila.actor AS a
LIMIT 10;


/* 
   16) WHERE vs HAVING (Clear difference demo)
   
   WHERE: row-level filter (before grouping)
   HAVING: group-level filter (after grouping)
    */

-- WHERE example: only payments > 5 are considered before grouping
SELECT customer_id, COUNT(*) AS num_payments_over_5
FROM sakila.payment
WHERE amount > 5
GROUP BY customer_id
ORDER BY num_payments_over_5 DESC;

-- HAVING example: all payments included, but groups filtered by SUM(amount)
SELECT customer_id, SUM(amount) AS total_payment
FROM sakila.payment
GROUP BY customer_id
HAVING SUM(amount) > 100
ORDER BY total_payment DESC;


/* 
   17) Order of execution in SQL (Important concept)
   
 -- SQL is written like:
-- SELECT ... FROM ... WHERE ... GROUP BY ... HAVING ... ORDER BY ... LIMIT ...
-- But executed roughly like:
-- FROM -> JOIN -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY -> LIMIT
    */


-- Find top 10 longest movies
SELECT film_id, title, length
FROM sakila.film
ORDER BY length DESC
LIMIT 10;

-- Count films per rating
SELECT rating, COUNT(*) AS films_in_rating
FROM sakila.film
GROUP BY rating
ORDER BY films_in_rating DESC;

-- Find cities containing "an" anywhere (pattern match)
SELECT city
FROM sakila.city
WHERE city LIKE '%an%';

-- Find customers who have exactly 30 rentals (HAVING exact match)
SELECT customer_id, COUNT(*) AS rental_count
FROM sakila.rental
GROUP BY customer_id
HAVING COUNT(*) = 30
ORDER BY customer_id;
