-- Assignment 1 By Ritesh Reddy Koya

USE sakila;

-- Q1) Get all customers whose first name starts with 'J' and who are active.

SELECT customer_id, first_name, last_name, active
FROM sakila.customer
WHERE first_name LIKE 'J%' AND active = 1;

/*
Explanation:
- LIKE 'J%' means first_name starts with J (% = any characters after).
- active = 1 filters active customers.
- WHERE filters rows before output.
*/


-- Q2) Find all films where the title contains the word 'ACTION' or the description contains 'WAR'.
 
SELECT film_id, title, description
FROM sakila.film
WHERE title LIKE '%ACTION%' OR description LIKE '%WAR%';

/*
Explanation:
- '%ACTION%' means ACTION anywhere in the title.
- OR means either condition can be true.
*/


-- Q3) List all customers whose last name is not 'SMITH'and whose first name ends with 'a'.

SELECT customer_id, first_name, last_name
FROM sakila.customer
WHERE last_name <> 'SMITH'
  AND first_name LIKE '%a';

/*
Explanation:
- <> means not equal.
- LIKE '%a' means ends with letter a.
- AND means both conditions must be true.
*/


-- Q4) Get all films where rental rate > 3.0 and replacement cost is not null.

SELECT film_id, title, rental_rate, replacement_cost
FROM sakila.film
WHERE rental_rate > 3.0
  AND replacement_cost IS NOT NULL;

/*
Explanation:
- IS NOT NULL is the correct way to check nulls.
- rental_rate > 3.0 filters expensive rentals.
*/


-- Q5) Count how many customers exist in each store who have active status = 1.

SELECT store_id, COUNT(*) AS active_customers
FROM sakila.customer
WHERE active = 1
GROUP BY store_id;

/*
Explanation:
- WHERE filters only active customers first.
- GROUP BY store_id groups customers by store.
- COUNT(*) counts rows in each group.
*/


-- Q6) Show distinct film ratings available in the film table.

SELECT DISTINCT rating
FROM sakila.film;

/*
Explanation:
- DISTINCT removes duplicates in output.
*/


-- Q7) Find the number of films for each rental duration where the average length is more than 100 minutes.

SELECT rental_duration,
       COUNT(*) AS film_count,
       AVG(length) AS avg_length
FROM sakila.film
GROUP BY rental_duration
HAVING AVG(length) > 100;

/*
Explanation:
- AVG(length) is an aggregate function.
- HAVING filters groups (after GROUP BY).
- WHERE cannot filter by AVG() because WHERE runs before grouping.
*/


-- Q8) List payment dates and total amount paid per date,but only include days where more than 100 payments were made.

SELECT DATE(payment_date) AS pay_date,
       SUM(amount) AS total_paid,
       COUNT(*) AS payment_count
FROM sakila.payment
GROUP BY DATE(payment_date)
HAVING COUNT(*) > 100
ORDER BY pay_date;

/*
Explanation:
- DATE(payment_date) removes time, keeping only the date.
- GROUP BY DATE(payment_date) groups all payments by day.
- HAVING COUNT(*) > 100 keeps only days with more than 100 payments.
*/


-- Q9) Find customers whose email is null OR ends with '.org'.

SELECT customer_id, first_name, last_name, email
FROM sakila.customer
WHERE email IS NULL OR email LIKE '%.org';

/*
Explanation:
- email IS NULL checks missing emails.
- LIKE '%.org' means email ends with .org
*/


-- Q10) List all films with rating 'PG' or 'G',order by rental rate descending.

SELECT film_id, title, rating, rental_rate
FROM sakila.film
WHERE rating IN ('PG', 'G')
ORDER BY rental_rate DESC;

/*
Explanation:
- IN ('PG','G') for OR condition.
- ORDER BY rental_rate DESC shows highest rental_rate first.
*/


-- Q11) Count how many films exist for each length where title starts with 'T' and count > 5.

SELECT length,
       COUNT(*) AS film_count
FROM sakila.film
WHERE title LIKE 'T%'
GROUP BY length
HAVING COUNT(*) > 5
ORDER BY film_count DESC;

/*
Explanation:
- WHERE title LIKE 'T%' filters titles starting with T.
- GROUP BY length groups by film length.
- HAVING COUNT(*) > 5 filters groups.
- did not get output because none of the title starting with T goes to count 5
*/


-- Q12) List all actors who have appeared in more than 10 films.

SELECT a.actor_id, a.first_name, a.last_name, fa_stats.film_count
FROM sakila.actor a,
     (
        SELECT actor_id, COUNT(film_id) AS film_count
        FROM sakila.film_actor
        GROUP BY actor_id
        HAVING COUNT(film_id) > 10
     ) fa_stats
WHERE a.actor_id = fa_stats.actor_id;


/*
Explanation:
- (subquery): compute film_count per actor_id from film_actor
  using GROUP BY + COUNT(film_id), then filter with HAVING > 10.
- (outer query): match those actor_id values to the actor table
  to show actor names + the film_count number.


JOIN version:
 SELECT a.actor_id, a.first_name, a.last_name, COUNT(fa.film_id) AS film_count
 FROM sakila.actor a
 JOIN sakila.film_actor fa ON a.actor_id = fa.actor_id
 GROUP BY a.actor_id, a.first_name, a.last_name
 HAVING COUNT(fa.film_id) > 10;
*/


-- Q13) Top 5 films with highest rental rates and longest lengths combined.Order by rental_rate first, length second.

SELECT film_id, title, rental_rate, length
FROM sakila.film
ORDER BY rental_rate DESC, length DESC
LIMIT 5;

/*
Explanation:
- ORDER BY rental_rate DESC sorts rental_rate high -> low.
- length DESC breaks ties by longest films first.
- LIMIT 5 returns top 5 rows.
*/


-- Q14) Show all customers along with total number of rentals,ordered most to least rentals.

SELECT
  c.customer_id,
  c.first_name,
  c.last_name,
  (
    SELECT COUNT(*)
    FROM sakila.rental r
    WHERE r.customer_id = c.customer_id
  ) AS total_rentals
FROM sakila.customer c
ORDER BY total_rentals DESC;

/*
Explanation:
- Correlated subquery runs once per customer row.
- It counts rentals in sakila.rental for that customer_id.
- ORDER BY total_rentals DESC shows most rentals first.

JOIN version:
 SELECT c.customer_id, c.first_name, c.last_name, COUNT(r.rental_id) AS total_rentals
 FROM sakila.customer c
 LEFT JOIN sakila.rental r ON c.customer_id = r.customer_id
 GROUP BY c.customer_id, c.first_name, c.last_name
 ORDER BY total_rentals DESC;
*/



-- Q15) List film titles that have never been rented.

SELECT film_id, title
FROM sakila.film
WHERE film_id NOT IN (
    SELECT DISTINCT i.film_id
    FROM sakila.inventory i
    WHERE i.inventory_id IN (
        SELECT r.inventory_id
        FROM sakila.rental r
    )
)
ORDER BY title;

/*
Explanation:
- Rentals happen through inventory copies:
    film -> inventory -> rental
- Inner-most query gets inventory_ids that were rented.
- Middle query maps those inventory_ids back to film_ids.
- Outer query selects films NOT IN that rented film list -> never rented.

JOIN version:
 SELECT f.film_id, f.title
 FROM sakila.film f
 LEFT JOIN sakila.inventory i ON f.film_id = i.film_id
 LEFT JOIN sakila.rental r ON i.inventory_id = r.inventory_id
 GROUP BY f.film_id, f.title
 HAVING COUNT(r.rental_id) = 0
 ORDER BY f.title;

*/
