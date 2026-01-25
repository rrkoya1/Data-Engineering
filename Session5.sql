-- Session 5, also Session 3,part 1

USE sakila;

-- 1) STRING FUNCTIONS

select title from sakila.film;

/* 
   1.1 LPAD + RPAD (padding on both sides)
   RPAD(title, 20, '*') pads the title on the RIGHT to length 20.
   LPAD(..., 25, '*') then pads that result on the LEFT to length 25.
   Useful for formatting output into fixed-width text.
  
    */
SELECT title, LPAD(RPAD(title, 20, '*'),25,'*') AS left_padded
FROM sakila.film
LIMIT 5;

/* 
   1.2 LPAD (left padding only)
   LPAD(title, 20, '*') pads the title on the LEFT to length 20.
 */
SELECT title, LPAD(title, 20, '*') AS left_padded
FROM sakila.film
LIMIT 5;


/* 
   1.3 SUBSTRING (extract a portion of text)
   SUBSTRING(title, 1, 9) takes 9 characters starting at position 1.
   Note: MySQL uses 1-based positions for substring.
  
    */
SELECT title, SUBSTRING(title, 1,9)  AS short_title
FROM sakila.film;


/* 
   1.4 CONCAT (concatenation)
   CONCAT joins strings together into a single output string.
   Example: first_name.last_name
    */
SELECT CONCAT(first_name, '.', last_name) AS full_name
FROM sakila.customer;


/* 
   1.5 REVERSE (reverse characters)
   REVERSE(title) flips the characters in the title.
 */
SELECT title, REVERSE(title) AS reversed_title
FROM sakila.film
LIMIT 5;


/* 
   1.6 LENGTH (string length)
   LENGTH(title) returns the number of BYTES in the string.
   For normal English titles, bytes usually equals characters.
   */
SELECT title, LENGTH(title) AS title_length
FROM sakila.film
WHERE LENGTH(title) =8;


/* 
   1.7 SUBSTRING with LOCATE (extract after a delimiter)
   LOCATE('@', email) finds the position of '@' in the email.
   SUBSTRING(email, LOCATE('@', email) + 1) returns everything after '@'.
   This extracts the domain part.
    */
select email from sakila.customer;

SELECT email,
       SUBSTRING(email, LOCATE('@', email) +1) AS domain
FROM sakila.customer;

/* 
   1.8 SUBSTRING + LOCATE + SUBSTRING_INDEX
   Takes the text after '@', then uses substring_index(..., '.', -1)
   to extract the LAST portion after the final '.'.
    */
SELECT 
  email,
  substring_index(SUBSTRING(email, LOCATE('@', email) + 1), '.', -1) AS domain
FROM 
  sakila.customer;

/* 
   1.9 SUBSTRING_INDEX (split by delimiter)
   substring_index(email,'@', 1) returns everything BEFORE '@'
   Useful for extracting usernames from emails.
    */
select substring_index(email,'@', 1) from sakila.customer;


/* 
   1.10 UPPER / LOWER + LIKE search
   UPPER() and LOWER() change letter case for consistent searching.
   - LIKE '%LOVELY%' means contains LOVELY
   - LIKE '%MAN' means ends with MAN
 */
SELECT title, UPPER(title),lower(title)
FROM sakila.film
WHERE UPPER(title) LIKE '%LOVELY%' or UPPER(title) LIKE '%MAN';

select title, lower(title) as lower_titles
FROM sakila.film;


/* 
   1.11 LEFT / RIGHT + GROUP BY + COUNT
   LEFT(title,2) gets first 2 characters.
   RIGHT(title,3) gets last 3 characters.
   GROUP BY groups films by these “prefix/suffix” patterns.
   COUNT(*) counts how many films fall in each group.
    */
SELECT LEFT(title, 2) AS first_letter, right(title,3) as last_letter,  COUNT(*) AS film_count
FROM sakila.film
GROUP BY LEFT(title, 2), right(title,3)
ORDER BY film_count DESC;

/* 
   1.12 LEFT / RIGHT (raw preview without grouping)
   Shows each film with its prefix/suffix extracted.
    */
SELECT LEFT(title,2) AS first_letter, right(title, 3) as last_letter, title 
from sakila.film;


/* 
   1.13 CASE WHEN + BETWEEN (bucketing)
   Categorizes customers into groups based on the first letter of last_name.
   BETWEEN 'A' AND 'M' works because letters are compared alphabetically.
 */
SELECT last_name,
       CASE 
           WHEN LEFT(last_name, 1) BETWEEN 'A' AND 'M' THEN 'Group A-M'
           WHEN LEFT(last_name, 1) BETWEEN 'N' AND 'Z' THEN 'Group N-Z'
           ELSE 'Other'
       END AS group_label
FROM sakila.customer;


/* 
   1.14 REPLACE (substitute characters)
   REPLACE(title, 'A', 'x') replaces A with x in the output.
*/
SELECT title, REPLACE(title, 'A', 'x') AS cleaned_title
FROM sakila.film
WHERE title LIKE '% %';


/* 
   1.15 REGEXP (pattern matching)
   REGEXP filters strings using regular expressions.
   Important symbols in your patterns:
   - [aeiouAEIOU]  => any vowel (upper/lower)
   - [^aeiouAEIOU] => any NON-vowel
   - {3}           => exactly 3 in a row
   - $             => end of string

   NOTE:
   last_name NOT REGEXP '[^aeiouAEIOU]{3}'
   This actually means: keep names that do NOT contain
   3 consecutive NON-vowels.
   
   example:
   - .      : any single character
   - ^      : start of string
   - $      : end of string
   - [abc]  : one of a,b,c
   - [a-z]  : range
   - [^a-z] : NOT in range
   - *      : repeat 0 or more
   - +      : repeat 1 or more
   - ?      : optional (0 or 1)
   - {m,n}  : between m and n repeats
   - |      : OR
    */

-- not contains 3 consecutive vowels 
SELECT customer_id, last_name
FROM sakila.customer
WHERE last_name NOT REGEXP '[^aeiouAEIOU]{3}'; 

-- ends with vowel
SELECT lower(title)
FROM sakila.film
WHERE title REGEXP '[aeiouAEIOU]$';

select title, right(title,2)
FROM sakila.film
WHERE title REGEXP '[eE]$' 
;

-- count 
select right(title,1), count(*)
FROM sakila.film
WHERE title REGEXP '[aeiouAEIOU]$' 
group by right(title,1)
;

SELECT title AS ending, right(title,1)
FROM sakila.film
WHERE title REGEXP '[Ee]$';

-- 2) MATH FUNCTIONS

/* 
   2.1 ^ operator 
   In MySQL, the ^ operator is BITWISE XOR (not exponent).
   MySQL will convert numbers to integers and apply XOR.
   That’s why it “allows” it — it’s a valid operator.

   So this is NOT “rental_rate cubed”.
   It is “rental_rate XOR 3”.
  */
SELECT title, rental_rate, rental_rate ^ 3 AS double_rate   -- debug why its allwoing string + integer
FROM sakila.film;

/* 
   2.2 Aggregation math on payments
   - COUNT(payment_id): number of payments (payment_id is PK, so not NULL)
   - SUM(amount): total paid
   - SUM(amount)/COUNT(payment_id): average payment
    */
SELECT customer_id,
       COUNT(payment_id) AS payments,
       SUM(amount) AS total_paid,
       SUM(amount) / COUNT(payment_id) AS avg_payment
       
FROM sakila.payment
GROUP BY customer_id;


/* 
   2.3 Selecting an added column (cost_efficiency_dup1)
    */
select rental_duration,cost_efficiency_dup1 from sakila.film;

select rental_duration from sakila.film;

/* 
   2.4  ALTER TABLE (MODIFIES SAKILA SCHEMA)
   Adds a new column cost_efficiency_dup1 to sakila.film.
 */
ALTER TABLE sakila.film
ADD COLUMN cost_efficiency_dup1 DECIMAL(6,2);

/* 
   2.5 Safe updates toggle
   Workbench may block updates; this disables that protection.
   */
SET SQL_SAFE_UPDATES = 0;

/* 
   2.6 UPDATE (MODIFIES SAKILA DATA)
   Fills cost_efficiency_dup1 for rows where length IS NOT NULL.
   This changes film table data.
    */
UPDATE sakila.film
SET cost_efficiency_dup1 = rental_duration * 2
WHERE length IS NOT NULL;

select * from sakila.film;


/* 
   2.7 RAND + FLOOR
   RAND() gives 0..1 decimal.
   FLOOR(RAND()*100) turns it into integer 0..99.
 */
SELECT customer_id, (RAND() * 100), FLOOR(RAND() * 100) AS random_score
FROM sakila.customer
LIMIT 5;


/* 
   2.8 POWER (exponent)
   POWER(rental_duration,2) squares rental_duration.
    */
SELECT film_id,rental_duration, POWER(rental_duration, 2) AS squared_duration
FROM sakila.film
LIMIT 5;


/* 
   2.9 MOD (remainder)
   MOD(length,60) gives leftover minutes if length is minutes.
*/
SELECT film_id,length, MOD(length, 60) AS minutes_over_hour
FROM sakila.film;


/* 
   2.10 CEIL / FLOOR rounding
   CEIL rounds up, FLOOR rounds down.
   */
SELECT rental_rate, CEIL(rental_rate) AS ceil_value, FLOOR(rental_rate) AS floor_value
FROM sakila.film;


/* 
   2.11 ROUND
   Rounds division result to 0 decimals and 1 decimal.
    */
SELECT rental_rate, ROUND(replacement_cost / rental_rate, 0),ROUND(replacement_cost / rental_rate, 1) AS ratio
FROM sakila.film;


-- 3) DATE / TIME FUNCTIONS


/* 
   3.1 DATEDIFF
   DATEDIFF(return_date, rental_date) gives days between dates.
   return_date must be NOT NULL for meaningful result.
   */
SELECT rental_id, return_date,rental_date, DATEDIFF(return_date, rental_date) AS days_rented
FROM sakila.rental
WHERE return_date IS NOT NULL;

/* 
   3.2 dayname / monthname
   Extracts readable day/month from last_update.
    */
select last_update,dayname(last_update),monthname(last_update) from sakila.film;

/* 
   3.3 year()
   Extracts year from rental_date.
 */
SELECT 
    rental_date, year(rental_date)
FROM
   sakila.rental;

SELECT payment_date FROM sakila.payment;

/* 
   3.4 DATE(payment_date) + SUM(amount)
   NOTE: GROUP BY DATE(payment_date), payment_date groups by both date and exact timestamp.
   This creates many small groups because payment_date includes time.
    */
SELECT payment_date, date(payment_date) AS pay_date, SUM(amount) AS total_paid
FROM sakila.payment
GROUP BY DATE(payment_date),payment_date
ORDER BY pay_date DESC;

#Find Customers Who Paid in the Last 24 Hours

select * from sakila.payment;

/* 
   3.5 NOW() - INTERVAL 1 DAY
   Filters payments within the last 24 hours from current time.
   */
SELECT customer_id, amount, payment_date
FROM sakila.payment
WHERE payment_date >= NOW() - INTERVAL 1 DAY;

select max(payment_date) FROM sakila.payment;

/* 
   3.6 Subquery with MAX(payment_date)
   Gets payments in the last 10 days relative to the newest payment.
    */
SELECT customer_id, amount, payment_date
FROM sakila.payment
WHERE payment_date >= (
    SELECT MAX(payment_date) - INTERVAL 10 day
    FROM sakila.payment
);

select now()  - INTERVAL 1 DAY as yesterday;

/* 
   3.7 CONCAT with CURDATE / NOW
   Creates readable output messages.
    */
SELECT CONCAT('Today is: ', CURDATE()) AS message;
SELECT CONCAT('Today is: ', now()) AS message;

SELECT NOW(), CURDATE(), CURRENT_TIME;


-- 4) CASTING (MODIFIES SAKILA)


/* 
   4.1  ALTER TABLE (MODIFIES SAKILA SCHEMA)
   Adds a new column amount_str into sakila.payment.
    */
ALTER TABLE sakila.payment
ADD COLUMN amount_str VARCHAR(20);

SET SQL_SAFE_UPDATES = 0;

/* 
   4.2  UPDATE (MODIFIES SAKILA DATA)
   Stores a string version of amount into amount_str.
   */
UPDATE sakila.payment
SET amount_str = CAST(amount AS CHAR);

select * from sakila.customer;
select * from sakila.payment;

/* 
   4.3 DROP COLUMN (MODIFIES SAKILA SCHEMA)
   Removes amount_str column.
    */
ALTER TABLE sakila.payment
drop COLUMN amount_str;

/* 
   4.4 Error already dropped amount_str;
    */
SELECT amount, amount_Str, amount + 10 AS numeric_add,
       amount_str + 10 AS string_add
FROM sakila.payment
LIMIT 5;


/* 
   4.5 SHOW COLUMNS + CAST date string to datetime
   SHOW COLUMNS displays table structure (metadata).
   CAST converts a string into datetime type.
   */
SHOW COLUMNS FROM sakila.payment;
SELECT CAST('2017-08-25' AS datetime);
