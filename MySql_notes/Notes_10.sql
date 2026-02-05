
-- SESSION 10 : INDEXING & FINE-TUNING


USE sakila;


-- 1) WHAT IS INDEXING?

/*
   Indexing:
   - An index is a data structure (B-Tree in MySQL).
   - It improves SELECT performance by avoiding full table scans.
   - Without index → MySQL scans every row.
   - With index → MySQL jumps directly to matching rows.

   Trade-off:
   - Faster SELECT
   - Slower INSERT / UPDATE / DELETE (index must be maintained)
*/

-- Without index → Full table scan
SELECT * 
FROM sakila.customer 
WHERE last_name = 'SMITH';


-- 2) PRIMARY KEY & CLUSTERED INDEX

/*
   In InnoDB:
   - PRIMARY KEY is the CLUSTERED INDEX
   - Data rows are physically stored in PRIMARY KEY order
   - Only ONE clustered index per table
*/

-- Fastest possible lookup (clustered index)
SELECT * 
FROM sakila.customer 
WHERE customer_id = 5;

-- 3) NON-CLUSTERED (SECONDARY) INDEX

/*
   Non-clustered index:
   - Separate structure from table data
   - Stores: indexed column + primary key reference
   - May require extra lookup to clustered index
*/

-- Check execution plan BEFORE indexing
EXPLAIN 
SELECT * 
FROM sakila.customer 
WHERE last_name = 'SMITH';

-- Add non-clustered index
CREATE INDEX idx_customer_last_name 
ON sakila.customer(last_name);

-- Check execution plan AFTER indexing
EXPLAIN 
SELECT * 
FROM sakila.customer 
WHERE last_name = 'SMITH';

-- 4) NATURAL KEY vs SURROGATE KEY

/*
   Natural Key:
   - Meaningful real-world data (SSN, Email)
   - Can change → bad for clustered index

   Surrogate Key:
   - Artificial (AUTO_INCREMENT)
   - Stable, small, efficient
   - Recommended as PRIMARY KEY
*/

-- Natural key example
CREATE TABLE sakila.employee_natural (
    ssn CHAR(11) PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50)
);

-- Duplicate natural key → ERROR
-- INSERT INTO sakila.employee_natural VALUES ('123-45-6789', 'Eve', 'HR');

-- Surrogate key example (recommended)
CREATE TABLE sakila.employee_surrogate (
    emp_id INT AUTO_INCREMENT PRIMARY KEY,
    ssn CHAR(11),
    name VARCHAR(100),
    department VARCHAR(50)
);

-- Same SSN allowed (surrogate key handles uniqueness)
INSERT INTO sakila.employee_surrogate (ssn, name, department)
VALUES ('123-45-6789', 'Alice', 'Finance'),
       ('123-45-6789', 'Bob', 'IT');
	   

-- OPTIMIZER REALITY CHECK
-- MySQL does NOT blindly use indexes.
-- It chooses execution plans based on estimated cost.
-- Sometimes a FULL TABLE SCAN is faster than using an index.
--
-- ALWAYS use EXPLAIN to understand query behavior.


-- 5) QUERY FINE-TUNING :

-- 1. Avoid SELECT *
SELECT first_name, last_name 
FROM sakila.customer;

-- 2. Use WHERE before GROUP BY and HAVING

SELECT store_id, COUNT(*) AS total_customers
FROM sakila.customer
WHERE active = 1
GROUP BY store_id
HAVING COUNT(*) > 200;

-- 3. Use JOIN instead of subquery

-- Subquery (less efficient)
SELECT first_name
FROM customer
WHERE store_id IN (
    SELECT store_id 
    FROM store 
    WHERE address_id = 1
);

-- JOIN (more efficient)
SELECT c.first_name
FROM customer c
JOIN store s ON c.store_id = s.store_id
WHERE s.address_id = 1;

-- 4. Avoid functions on indexed columns


-- BAD: index not used
EXPLAIN 
SELECT * 
FROM rental 
WHERE YEAR(rental_date) = 2005;

-- GOOD: index preserved
EXPLAIN 
SELECT * 
FROM rental 
WHERE rental_date BETWEEN '2005-01-01' AND '2005-12-31';

-- 5. Use LIMIT effectively
SELECT *
FROM film
ORDER BY film_id
LIMIT 1000;

-- 6. Use EXPLAIN to understand execution plan

EXPLAIN 
SELECT *
FROM customer
WHERE store_id = 1;

-- 7. Use CTE for readable query breakdown

WITH high_paying_customers AS (
    SELECT customer_id, SUM(amount) AS total_paid
    FROM payment
    GROUP BY customer_id
    HAVING SUM(amount) > 100
)
SELECT c.first_name, c.last_name, h.total_paid
FROM customer c
JOIN high_paying_customers h
ON c.customer_id = h.customer_id;

-- 8. Maintain database statistics

ANALYZE TABLE customer;
OPTIMIZE TABLE rental;

-- 9. Avoid large OFFSET in pagination

-- Inefficient
SELECT *
FROM payment
LIMIT 1000, 10;

-- Efficient
SELECT *
FROM payment
WHERE payment_id > 1000
LIMIT 10;

-- 10. Index only useful columns
/*
   - High cardinality columns → GOOD for indexing
   - Low cardinality (gender, status) → usually BAD
*/


DROP INDEX idx_customer_last_name ON sakila.customer;

--Question discussed in class

-- Q1) What happens if a column is NOT indexed?
-- ANSWER:
-- MySQL performs a FULL TABLE SCAN.
-- It reads every row one-by-one to find matching data.
--
-- WHY:
-- Time complexity becomes O(n).
-- This is the primary cause of slow queries on large tables.


-- Q2) Does an index store the actual table data?
-- ANSWER:
-- NO.
-- An index stores:
--   - Indexed column value
--   - Reference (pointer) to the actual row
--
-- WHY:
-- Smaller data structure = faster searches.
-- Index acts as a reference point, not a duplicate table.


-- Q3) Why is the Primary Key called a Clustered Index in MySQL?
-- ANSWER:
-- In InnoDB, the PRIMARY KEY defines the PHYSICAL order of rows.
--
-- WHY:
-- Data is stored in Primary Key order on disk.
-- Only ONE clustered index is possible per table.


-- Q4) Can a table have multiple clustered indexes?
-- ANSWER:
-- NO.
--
-- WHY:
-- A table can be physically ordered in only ONE way.
-- All other indexes are non-clustered (secondary).


-- Q5) What happens internally when using a non-clustered index?
-- ANSWER:
-- Step 1: MySQL searches the secondary index.
-- Step 2: Retrieves the Primary Key value.
-- Step 3: Uses the Primary Key to fetch the full row.
--
-- This process is called:
-- "Bookmark Lookup" / "Back-to-table lookup"
--
-- WHY:
-- Extra lookup makes secondary index slower than PK lookup.


-- Q6) Do indexes always improve performance?
-- ANSWER:
-- NO.
--
-- WHY:
-- Indexes speed up SELECT queries,
-- but slow down:
--   INSERT
--   UPDATE
--   DELETE
--
-- Because MySQL must update index structures every time data changes.


-- Q7) What is the difference between Natural Key and Surrogate Key?
-- NATURAL KEY:
-- - Real-world meaning (Email, SSN, Phone)
-- - Can change
-- - Larger size
-- - Not ideal for clustered index
--
-- SURROGATE KEY:
-- - Artificial (AUTO_INCREMENT)
-- - Small, numeric, stable
-- - Best choice for Primary Key


-- Q8) Are all Primary Keys Natural Keys?
-- ANSWER:
-- NO.
--
-- Most modern databases use SURROGATE keys as Primary Keys.


-- Q9) Are all Natural Keys Primary Keys?
-- ANSWER:
-- NO.
--
-- Natural keys are often enforced using UNIQUE constraints,
-- not Primary Keys.


-- Q10) Why does MySQL sometimes ignore an index even when it exists?
-- ANSWER:
-- The Optimizer decides the index is NOT cost-effective.
--
-- Common reasons:
-- 1) Low cardinality (many duplicate values)
-- 2) Function used on indexed column
-- 3) Outdated table statistics
--
-- FIX:
ANALYZE TABLE sakila.customer;


