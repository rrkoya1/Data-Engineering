-- Create database
CREATE DATABASE company_db;
USE company_db;

-- Create Table
CREATE TABLE test_table ( id INT, name VARCHAR(100));
INSERT INTO test_table (id, name)
VALUES
(1, 'Alice'),
(2, 'BoB'),
(3, 'Charlie');
INSERT INTO test_table (id, name)
VALUES (5,5); -- allowed but bad practice/info

SELECT * FROM test_table;
ALTER TABLE test_table
ADD Email VARCHAR(255); -- add column
ALTER TABLE test_table
RENAME COLUMN Email TO email_id; -- rename

DROP TABLE IF EXISTS Persons;
CREATE TABLE Persons (
ID INT NOT NULL UNIQUE,
LastName VARCHAR(255) NOT NULL,
FirstName VARCHAR(255),
Age int
);
INSERT INTO Persons VALUES (1, 'Smith', 'John', 30);
INSERT INTO Persons VALUES (2, 'Doe', NULL, NULL);

INSERT INTO Persons VALUES (1, 'Brown', 'Charlie', 25); -- error duplicate

INSERT INTO Persons VALUES (3, NULL, 'Alice', 28); -- error null voilation

ALTER TABLE Persons ADD PRIMARY KEY (ID); -- primary key

SELECT CONSTRAINT_NAME, CONSTRAINT_TYPE
FROM information_schema.TABLE_CONSTRAINTS
WHERE TABLE_SCHEMA = 'company_db'
AND TABLE_NAME = 'Persons';
ALTER TABLE Persons DROP PRIMARY KEY;
ALTER TABLE Persons
ADD CONSTRAINT PK_Person PRIMARY KEY (ID);

-- child table
CREATE TABLE Orders (
    OrderID INT PRIMARY KEY,
    OrderDate DATE,
    PersonID INT,
    FOREIGN KEY (PersonID) REFERENCES Persons(ID) -- foreign key
        ON DELETE RESTRICT
        ON UPDATE CASCADE
);
INSERT INTO Orders VALUES (1001, '2024-06-10', 1);

INSERT INTO Orders VALUES (1002, '2024-06-11', 999); -- invalid insert

DELETE FROM Persons WHERE ID = 1; -- error restrict

UPDATE Persons SET ID =4 WHERE ID = 1; -- cascade allowed

-- check and default constraints
CREATE TABLE employee (
    ID INT NOT NULL,
    LastName VARCHAR(255) NOT NULL,
    FirstName VARCHAR(255),
    Age INT CHECK (Age >= 18),
    city VARCHAR(255) DEFAULT 'New York'
);
INSERT INTO employee (ID, LastName, FirstName, Age)
VALUES (4, 'Joey', 'Tribiani', 21);
SET SQL_SAFE_UPDATES = 0;

DELETE FROM test_table WHERE id = 1; -- delete

TRUNCATE TABLE test_table; -- truncate

DROP TABLE test_table; -- drop

CREATE INDEX idx_lastname ON Persons(LastName); --index

CREATE VIEW adult_persons AS -- view
SELECT ID, FirstName, LastName
FROM Persons
WHERE Age >= 18;
SELECT * FROM adult_persons;

SELECT p.FirstName, o.OrderID, o.OrderDate --join
FROM Persons p
JOIN Orders o
ON p.ID = o.PersonID;

START TRANSACTION; -- transactions
INSERT INTO Persons VALUES (10, 'Test', 'User', 25);
ROLLBACK;  
COMMIT;    
 
DROP TABLE Orders;
DROP TABLE Persons;
DROP TABLE employee;
DROP DATABASE company_db;