-- ASSIGNMENT 3, by Ritesh Reddy Koya

USE sakila;


-- Q1) Display all customer details who have made more than 5 payments.

/*
Answer:
- First, count payments per customer in sakila.payment.
- Keep only customers with COUNT(payment_id) > 5.
- Then return full customer rows from sakila.customer for those customer_id values.
*/

SELECT *
FROM sakila.customer
WHERE customer_id IN (
    SELECT customer_id
    FROM sakila.payment
    GROUP BY customer_id
    HAVING COUNT(payment_id) > 5
);

/* JOIN :
SELECT c.*, p_stats.payment_count
FROM sakila.customer c
JOIN (
    SELECT customer_id, COUNT(payment_id) AS payment_count
    FROM sakila.payment
    GROUP BY customer_id
    HAVING COUNT(payment_id) > 5
) p_stats ON c.customer_id = p_stats.customer_id;
*/

-- Q2) Find the names of actors who have acted in more than 10 films.

/*
Answer:
- sakila.film_actor links actors to films.
- Group by actor_id and count how many films each actor appears in.
- Keep only actors with COUNT(film_id) > 10, then show their names from sakila.actor.
*/

SELECT actor_id, first_name, last_name
FROM sakila.actor
WHERE actor_id IN (
    SELECT actor_id
    FROM sakila.film_actor
    GROUP BY actor_id
    HAVING COUNT(film_id) > 10
);

/* JOIN:
SELECT a.actor_id, a.first_name, a.last_name, fa_stats.film_count
FROM sakila.actor a
JOIN (
    SELECT actor_id, COUNT(film_id) AS film_count
    FROM sakila.film_actor
    GROUP BY actor_id
    HAVING COUNT(film_id) > 10
) fa_stats ON a.actor_id = fa_stats.actor_id;
*/


-- Q3) Find the names of customers who never made a payment.
/*
Answer:
- Find all customer_id values that appear in sakila.payment.
- Return customers whose customer_id is NOT in that list.
*/

SELECT customer_id, first_name, last_name
FROM sakila.customer
WHERE customer_id NOT IN (
    SELECT DISTINCT customer_id
    FROM sakila.payment
);

/* JOIN:
SELECT c.customer_id, c.first_name, c.last_name
FROM sakila.customer c
LEFT JOIN sakila.payment p ON c.customer_id = p.customer_id
WHERE p.payment_id IS NULL;
*/

-- Q4) List all films whose rental rate is higher than the average rental rate of all films.
/*
Answer:
- Compute the overall AVG(rental_rate) from sakila.film.
- Return only films with rental_rate > that average.
*/
SELECT film_id, title, rental_rate
FROM sakila.film
WHERE rental_rate > (
    SELECT AVG(rental_rate)
    FROM sakila.film
)
ORDER BY rental_rate DESC;

-- Q5) List the titles of films that were never rented.
/*
Answer:
- A film is “rented” only if at least one of its inventory copies appears in sakila.rental.
- Inner subquery: rental.inventory_id list = all inventory items ever rented.
- Middle subquery: inventory rows whose inventory_id is in that rental list → gives film_id values that have been rented.
- Outer query: film rows whose film_id is NOT in that “rented film_id” list → never rented.
*/
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

/* JOIN:
SELECT f.film_id, f.title
FROM sakila.film f
LEFT JOIN sakila.inventory i ON f.film_id = i.film_id
LEFT JOIN sakila.rental r ON i.inventory_id = r.inventory_id
WHERE r.rental_id IS NULL
ORDER BY f.title;
*/


-- Q6) Display the customers who rented films in the same month as customer with ID 5.
/*
Answer:
- First find the (YEAR, MONTH) pair(s) when customer 5 rented.
- Then return all customers who have rentals in any of those same (YEAR, MONTH) pairs.
*/
SELECT customer_id, first_name, last_name
FROM sakila.customer
WHERE customer_id IN (
    SELECT DISTINCT customer_id
    FROM sakila.rental
    WHERE (YEAR(rental_date), MONTH(rental_date)) IN (
        SELECT YEAR(rental_date), MONTH(rental_date)
        FROM sakila.rental
        WHERE customer_id = 5
    )
);


/* JOIN:
SELECT DISTINCT c.customer_id, c.first_name, c.last_name
FROM sakila.customer c
JOIN sakila.rental r ON c.customer_id = r.customer_id
WHERE (YEAR(r.rental_date), MONTH(r.rental_date)) IN (
    SELECT YEAR(rental_date), MONTH(rental_date)
    FROM sakila.rental
    WHERE customer_id = 5
);
*/


-- Q7) Find all staff members who handled a payment greater than the average payment amount.
/*
Answer:
- Compute average payment amount across all payments.
- Pick staff_id values where at least one payment amount is greater than that average.
- GROUP BY staff_id removes duplicates.
*/
SELECT staff_id
FROM sakila.payment
WHERE amount > (
    SELECT AVG(amount)
    FROM sakila.payment
)
GROUP BY staff_id;

/* JOIN:
SELECT DISTINCT s.staff_id, s.first_name, s.last_name
FROM sakila.staff s
JOIN sakila.payment p ON s.staff_id = p.staff_id
WHERE p.amount > (SELECT AVG(amount) FROM sakila.payment);
*/

-- Q8) Show the title and rental duration of films whose rental duration is greater than the average.
/*
Answer:
- Compute AVG(rental_duration) from sakila.film.
- Return films with rental_duration greater than that average.
*/
SELECT film_id, title, rental_duration
FROM sakila.film
WHERE rental_duration > (
    SELECT AVG(rental_duration)
    FROM sakila.film
)
ORDER BY rental_duration DESC, title;


-- Q9) Find all customers who have the same address as customer with ID 1.
/*
Answer:
- Get address_id of customer 1 using a subquery.
- Return all customers whose address_id matches that value.
NOTE:
- This will include customer_id = 1 as well (because they match their own address).
*/
SELECT customer_id, first_name, last_name, address_id
FROM sakila.customer
WHERE address_id = (
    SELECT address_id
    FROM sakila.customer
    WHERE customer_id = 1
);

/* JOIN:
SELECT c2.customer_id, c2.first_name, c2.last_name, c2.address_id
FROM sakila.customer c1
JOIN sakila.customer c2 ON c1.address_id = c2.address_id
WHERE c1.customer_id = 1;
*/

-- Q10) List all payments that are greater than the average of all payments.
/*
Answer:
- Compute AVG(amount) across all sakila.payment.
- Return payment rows where amount > that average.
*/
SELECT payment_id, customer_id, staff_id, amount, payment_date
FROM sakila.payment
WHERE amount > (
    SELECT AVG(amount)
    FROM sakila.payment
)
ORDER BY amount DESC, payment_date DESC;

