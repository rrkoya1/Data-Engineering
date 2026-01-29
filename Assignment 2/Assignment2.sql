-- Assignment 2 by Rites Reddy Koya


USE sakila;


-- Q1) Identify if there are duplicates in Customer table.Don't use customer_id to check duplicates.

/* Answer:
   If two (or more) rows share the same "identity fields", that’s a duplicate.
   Since we can't use customer_id, we pick a reasonable combination:
   first_name + last_name + email + store_id + address_id.
*/
SELECT
  first_name, last_name, email, store_id, address_id,
  COUNT(*) AS duplicate_count
FROM sakila.customer
GROUP BY first_name, last_name, email, store_id, address_id
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;


-- Q2) Number of times letter 'a' is repeated in film descriptions

/* Answer:
   Count occurrences of 'a' by comparing string lengths:
   occurrences = LENGTH(text) - LENGTH(REPLACE(text, 'a', ''))
   We use LOWER() so 'A' and 'a' are both counted.
*/
SELECT
  SUM(
    LENGTH(LOWER(description)) - LENGTH(REPLACE(LOWER(description), 'a', ''))
  ) AS total_a_count
FROM sakila.film;

-- Per film in decresing order

SELECT title, 
       (CHAR_LENGTH(description) - CHAR_LENGTH(REPLACE(LOWER(description), 'a', ''))) AS a_count
FROM sakila.film
ORDER BY a_count DESC;

-- Q3) Number of times each vowel is repeated in film descriptions

/* Answer replace vowels with '', find the difference between length. */
SELECT
  SUM(LENGTH(LOWER(description)) - LENGTH(REPLACE(LOWER(description), 'a', ''))) AS a_count,
  SUM(LENGTH(LOWER(description)) - LENGTH(REPLACE(LOWER(description), 'e', ''))) AS e_count,
  SUM(LENGTH(LOWER(description)) - LENGTH(REPLACE(LOWER(description), 'i', ''))) AS i_count,
  SUM(LENGTH(LOWER(description)) - LENGTH(REPLACE(LOWER(description), 'o', ''))) AS o_count,
  SUM(LENGTH(LOWER(description)) - LENGTH(REPLACE(LOWER(description), 'u', ''))) AS u_count
FROM sakila.film;

--Per film

SELECT
  film_id,
  title,
  (CHAR_LENGTH(description) - CHAR_LENGTH(REPLACE(LOWER(description), 'a', ''))) AS a_count,
  (CHAR_LENGTH(description) - CHAR_LENGTH(REPLACE(LOWER(description), 'e', ''))) AS e_count,
  (CHAR_LENGTH(description) - CHAR_LENGTH(REPLACE(LOWER(description), 'i', ''))) AS i_count,
  (CHAR_LENGTH(description) - CHAR_LENGTH(REPLACE(LOWER(description), 'o', ''))) AS o_count,
  (CHAR_LENGTH(description) - CHAR_LENGTH(REPLACE(LOWER(description), 'u', ''))) AS u_count
FROM sakila.film
ORDER BY title;

-- Q4) Display the payments made by each customer: 1) Month wise  2) Year wise  3) Week wise

-- 4.1 Month-wise (by customer_id, year, month)
SELECT
  customer_id,
  YEAR(payment_date)  AS pay_year,
  MONTHNAME(payment_date) AS pay_month,
  COUNT(*)            AS payment_count,
  SUM(amount)         AS total_paid
FROM sakila.payment
GROUP BY customer_id, YEAR(payment_date), MONTHNAME(payment_date)
ORDER BY customer_id, pay_year, pay_month;


-- 4.2 Year-wise
SELECT
  customer_id,
  YEAR(payment_date) AS pay_year,
  COUNT(*)           AS payment_count,
  SUM(amount)        AS total_paid
FROM sakila.payment
GROUP BY customer_id, YEAR(payment_date)
ORDER BY customer_id, pay_year;

-- 4.3 Week-wise
-- WEEK(payment_date) gives a week number (0–53 ).

SELECT
  customer_id,
  YEAR(payment_date) AS pay_year,
  WEEK(payment_date) AS pay_week,
  COUNT(*)           AS payment_count,
  SUM(amount)        AS total_paid
FROM sakila.payment
GROUP BY customer_id, YEAR(payment_date), WEEK(payment_date)
ORDER BY customer_id, pay_year, pay_week;

-- Q5) Check if a given year is a leap year or not (no Sakila tables).Write within SELECT query with hardcoded date/year.
/*
Answer: we will check if february has 29 days for that year
*/
  SELECT
  2026 AS given_year,
  CASE
    WHEN DAY(LAST_DAY('2026-02-01')) = 29 THEN 'Leap Year'
    ELSE 'Not a Leap Year'
  END AS leap_year_result;

-- Q6) Display number of days remaining in the current year from today.
/* Find last day of current year, then DATEDIFF(last_day, today). */
SELECT
  CURDATE() AS today,
  (MAKEDATE(YEAR(CURDATE()), 1) + INTERVAL 1 YEAR - INTERVAL 1 DAY) AS last_day_of_year,
  DATEDIFF(
    (MAKEDATE(YEAR(CURDATE()), 1) + INTERVAL 1 YEAR - INTERVAL 1 DAY),
    CURDATE()
  ) AS days_remaining_in_year;


-- Q7) Display quarter number (Q1,Q2,Q3,Q4) for payment dates.

/* Answer: Quater () function, concate it to payment */
SELECT
  payment_id,
  payment_date,
  CONCAT('Q', QUARTER(payment_date)) AS quarter_label
FROM sakila.payment;

-- Q8) Display age in years, months, days based on your date of birth. Example: "21 years, 4 months, 12 days"
/* Answer:
   years  = TIMESTAMPDIFF(YEAR, dob, today)
   then we "move forward" dob by that many years,
   months = TIMESTAMPDIFF(MONTH, dob_plus_years, today)
   then move forward again by those months,
   days   = DATEDIFF(today, dob_plus_years_plus_months)
*/
SELECT
  '2000-09-01' AS dob,
  CONCAT(
    TIMESTAMPDIFF(YEAR, '2000-09-01', CURDATE()), ' years, ',
    TIMESTAMPDIFF(
      MONTH,
      DATE_ADD('2000-09-01', INTERVAL TIMESTAMPDIFF(YEAR, '2000-09-01', CURDATE()) YEAR),
      CURDATE()
    ), ' months, ',
    DATEDIFF(
      CURDATE(),
      DATE_ADD(
        DATE_ADD('2000-09-01', INTERVAL TIMESTAMPDIFF(YEAR, '2000-09-01', CURDATE()) YEAR),
        INTERVAL TIMESTAMPDIFF(
          MONTH,
          DATE_ADD('2000-09-01', INTERVAL TIMESTAMPDIFF(YEAR, '2000-09-01', CURDATE()) YEAR),
          CURDATE()
        ) MONTH
      )
    ), ' days'
  ) AS age_ymd;
