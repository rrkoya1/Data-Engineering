-- ASSIGNMENT 4 — by Ritesh Reddy koya


USE sakila;

-- Q1) List all customers along with the films they have rented.
/* Answer:
   customer -> rental -> inventory -> film
   Each rental points to an inventory copy, inventory points to film.
*/
SELECT
  c.customer_id,
  c.first_name,
  c.last_name,
  r.rental_id,
  r.rental_date,
  f.film_id,
  f.title
FROM sakila.customer c
JOIN sakila.rental r
  ON c.customer_id = r.customer_id
JOIN sakila.inventory i
  ON r.inventory_id = i.inventory_id
JOIN sakila.film f
  ON i.film_id = f.film_id
ORDER BY c.customer_id, r.rental_date;


-- Q2) List all customers and show their rental count,including those who haven't rented any films.

/* Answer:
   LEFT JOIN keeps ALL customers.
   COUNT(r.rental_id) counts only matching rentals.
*/
SELECT
  c.customer_id,
  c.first_name,
  c.last_name,
  COUNT(r.rental_id) AS rental_count
FROM sakila.customer c
LEFT JOIN sakila.rental r
  ON c.customer_id = r.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY rental_count DESC, c.customer_id;


-- Q3) Show all films along with their category.Include films that don't have a category assigned.

/* Answer:
   film -> film_category -> category
   LEFT JOIN keeps all films, even if no category mapping exists.
*/
SELECT
  f.film_id,
  f.title,
  c.name AS category
FROM sakila.film f
LEFT JOIN sakila.film_category fc
  ON f.film_id = fc.film_id
LEFT JOIN sakila.category c
  ON fc.category_id = c.category_id
ORDER BY f.title;

-- Q4) Show all customer and staff emails from both table using a FULL OUTER JOIN simulation (LEFT + RIGHT + UNION).

/* Answer:
   MySQL doesn't support FULL OUTER JOIN directly.
   We simulate it by:
   1) customer LEFT JOIN staff (never matches) to output customers
   2) staff RIGHT JOIN customer (never matches) to output staff
   UNION merges both sets (removes duplicates).
*/
-- 1. All Customers and their matching Staff (if any)
SELECT c.email AS customer_email, s.email AS staff_email
FROM sakila.customer c
LEFT JOIN sakila.staff s ON c.email = s.email

UNION

-- 2. All Staff and their matching Customers (if any)
SELECT c.email AS customer_email, s.email AS staff_email
FROM sakila.customer c
RIGHT JOIN sakila.staff s ON c.email = s.email
ORDER BY staff_email DESC;

/*
The staff who rented the customer:
 
SELECT DISTINCT
    c.email AS customer_email,
    s.email AS staff_email
FROM sakila.customer c
JOIN sakila.payment p ON c.customer_id = p.customer_id
JOIN sakila.staff s ON p.staff_id = s.staff_id;
*/

-- Q5) Find all actors who acted in the film "ACADEMY DINOSAUR".

/* Answer:
   film -> film_actor -> actor
   Filter by film title.
*/
SELECT
  a.actor_id,
  a.first_name,
  a.last_name
FROM sakila.film f
JOIN sakila.film_actor fa
  ON f.film_id = fa.film_id
JOIN sakila.actor a
  ON fa.actor_id = a.actor_id
WHERE f.title = 'ACADEMY DINOSAUR'
ORDER BY a.last_name, a.first_name;


-- Q6) List all stores and the total number of staff members working in each store, even if a store has no staff.

/* Answer:
   store LEFT JOIN staff keeps all stores.
   COUNT(staff_id) counts matching staff rows (NULL ignored).
*/
SELECT
  s.store_id,
  COUNT(st.staff_id) AS staff_count
FROM sakila.store s
LEFT JOIN sakila.staff st
  ON s.store_id = st.store_id
GROUP BY s.store_id
ORDER BY s.store_id;

-- Q7) List the customers who have rented films more than 5 times. Include their name and total rental count.
/* Answer:
   JOIN customer -> rental, then group and HAVING for threshold.
*/
SELECT
  c.customer_id,
  c.first_name,
  c.last_name,
  COUNT(r.rental_id) AS total_rentals
FROM sakila.customer c
JOIN sakila.rental r
  ON c.customer_id = r.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING COUNT(r.rental_id) > 5
ORDER BY total_rentals DESC, c.customer_id;
