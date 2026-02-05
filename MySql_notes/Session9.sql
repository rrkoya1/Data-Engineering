
-- Session 9 Notes — STORED PROCEDURES


USE sakila;


-- 1) STORED PROCEDURES — CONCEPT


/* 
   What is a Stored Procedure?
   - A stored procedure is a saved “program” inside the database.
   - It can contain multiple SQL statements.
   - It can accept inputs (parameters) and return output values.

   Why do we use stored procedures?
    Reusability: write once, call anytime
    Security: users can run procedure without direct table access
    Performance: compiled/optimized execution plan (often faster)
    Automation: can implement business logic inside DB

   Think of it like a SQL function, but more powerful.
    */



-- 2) DELIMITER — WHY IT IS NEEDED


/* 
   What is DELIMITER?
   - MySQL normally ends commands with semicolon: ;
   - Stored procedures contain many SQL statements, each ending with ;
   - So MySQL gets confused and ends early.

   Solution:
   - Change delimiter temporarily, e.g. to //
   - Write procedure code using ;
   - End procedure with // instead of ;

   Syntax:
     DELIMITER //
     CREATE PROCEDURE ...
     BEGIN
        ...
     END//
     DELIMITER ;
    */


-- 3) IN PARAMETER STORED PROCEDURE


/* 
   IN parameter means:
   - input into the procedure
   - procedure reads it but does not modify it

   Example use case:
   - “Show payments for a customer”
   - cid is the input value.
    */

DROP PROCEDURE IF EXISTS sakila.GetCustomerPayments;
DELIMITER //

-- IN parameter only
CREATE PROCEDURE sakila.GetCustomerPayments(IN cid INT)
BEGIN
    SELECT payment_id, amount, payment_date
    FROM sakila.payment
    WHERE customer_id = cid;
END;
//

DELIMITER ;

-- CALL example:
CALL sakila.GetCustomerPayments(7);

-- Explanation:
-- - cid = 7 goes into the procedure
-- - procedure returns payment rows for customer 7
-- - This does NOT change any data (safe query)



-- 4) OUT PARAMETER STORED PROCEDURE


/* 
   OUT parameter means:
   - output value returned by the procedure
   - very useful when you want “a computed result”
   - example: total amount paid by customer

   OUT params behave like a variable you pass into the procedure,
   and the procedure sets its value.
    */

DROP PROCEDURE IF EXISTS sakila.TotalPaid;
DELIMITER //

-- OUT parameter
CREATE PROCEDURE sakila.TotalPaid(IN cid INT, OUT total DECIMAL(10,2))
BEGIN
    SELECT SUM(amount) INTO total
    FROM sakila.payment
    WHERE customer_id = cid ;
END;
//

DELIMITER ;

-- CALL example:
CALL sakila.TotalPaid(6, @total);
SELECT @total;

-- Explanation:
-- - IN cid: customer id
-- - OUT total: stores SUM(amount)
-- - SELECT ... INTO total puts the result into output parameter


-- 5) DYNAMIC SQL PROCEDURE

/* 
   What is Dynamic SQL?
   - SQL where the table name / column name / query changes dynamically.
   - Stored procedures allow us to generate SQL as a string and execute it.

   Why needed?
   - Normal SQL does NOT allow table names to be parameters directly.
   - Example: SELECT COUNT(*) FROM tbl_name;  (does not work)

   So we build a query string and run it using:
   - PREPARE
   - EXECUTE
   - DEALLOCATE PREPARE

   Caution:
   Dynamic SQL can be dangerous if user input is not trusted
   (SQL Injection). In real projects, validate inputs carefully.
    */

DROP PROCEDURE IF EXISTS sakila.DynamicQuery;
DELIMITER //

-- Dynamic SQL procedure
CREATE PROCEDURE sakila.DynamicQuery(IN tbl_name VARCHAR(64))
BEGIN
    SET @s = CONCAT('SELECT COUNT(*) AS total_rows FROM ', tbl_name);
    PREPARE stmt FROM @s;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
END;
//

DELIMITER ;

-- CALL example:
CALL sakila.DynamicQuery('sakila.customer');

-- Explanation:
-- - If input = sakila.customer
-- - procedure builds query:
--     SELECT COUNT(*) AS total_rows FROM sakila.customer
-- - then executes it dynamically


-- 6) DYNAMIC PARAMETER (INOUT) — IN + OUT COMBINATION


/* 
   What is a Dynamic Parameter (INOUT)?
   - INOUT parameter works as BOTH input and output.
   - You send a starting value into the procedure (IN behavior),
     and procedure modifies it and returns the updated value (OUT behavior).

   Think of INOUT like:
   “Take a value → process it → give the updated value back”

   Why useful?
   When procedure needs to update a variable
   When you want running totals / counters
   When you want output that depends on the initial input

   Syntax:
     CREATE PROCEDURE name(INOUT param datatype)
     BEGIN
         SET param = param + 1;
     END;

   Calling:
     SET @x = 5;
     CALL procedure(@x);
     SELECT @x;
    */

DROP PROCEDURE IF EXISTS sakila.IncrementRentals;
DELIMITER //

CREATE PROCEDURE sakila.IncrementRentals(IN cid INT, INOUT rents INT)
BEGIN
    -- Add the rental count of the given customer into the variable rents
    SELECT COUNT(*) + rents INTO rents
    FROM sakila.rental
    WHERE customer_id = cid;
END;
//

DELIMITER ;

-- CALL example:
SET @rents = 0;
CALL sakila.IncrementRentals(3, @rents);
SELECT @rents;

-- Explanation:
-- - cid is IN: customer_id input
-- - rents is INOUT: starts with @rents value (0)
-- - procedure adds the rental count to it
-- - updated result comes back into @rents


-- 7) INFORMATION_SCHEMA — DATABASE METADATA

/* 
   What is information_schema?
   - It's a system database that stores “metadata” about databases.
   - Metadata = data about data.

   Examples:
   - list all tables in a schema
   - list columns of a table
   - check constraints, indexes, procedures, etc.

   Useful for:
    automation
    dynamic reporting
    admin queries
    */


-- 8) Example program


/* 
   Goal:
   - For every table in a schema (sakila), generate:
       SELECT count(*) FROM schema.table;
   - Store these generated SELECT queries into a temporary table.

   Concepts used:
   Temporary table
   Stored procedure
   Cursor (loop table-by-table)
   information_schema.tables
   Dynamic SQL using PREPARE/EXECUTE
    */


-- Create a temp table to store SELECT statements
DROP TEMPORARY TABLE IF EXISTS sakila.select_statements; 

CREATE TEMPORARY TABLE sakila.select_statements (
    id INT AUTO_INCREMENT PRIMARY KEY,
    statement_text TEXT
);

-- drop PROCEDURE sakila.StoreSelectStatements;
DELIMITER //

CREATE PROCEDURE sakila.StoreSelectStatements(IN db_name VARCHAR(64))
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE tbl_name VARCHAR(64);

    DECLARE cur CURSOR FOR
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = db_name;

    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

    OPEN cur;

    read_loop: LOOP
        FETCH cur INTO tbl_name;
        IF done THEN
            LEAVE read_loop;
        END IF;

        SET @stmt = CONCAT('SELECT count(*) FROM ', db_name, '.', tbl_name, ';');
        SET @ins = CONCAT('INSERT INTO select_statements (statement_text) VALUES (?)');
        PREPARE stmt FROM @ins;
        EXECUTE stmt USING @stmt;
        DEALLOCATE PREPARE stmt;

    END LOOP;

    CLOSE cur;
END;
//

DELIMITER ;

-- Call the procedure
CALL sakila.StoreSelectStatements('sakila');

-- See results
SELECT * FROM sakila.select_statements;

-- Explanation:
-- 1) Cursor fetches table_name one-by-one from information_schema.tables
-- 2) For each table, builds query like:
--       SELECT count(*) FROM sakila.actor;
-- 3) Inserts that query string into select_statements table
-- 4) End result = a list of count queries for each table
