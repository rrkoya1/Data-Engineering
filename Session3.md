# Session 3 — MySQL 1

## Types of Data
Relational databases are designed mainly for structured data.
* **Structured Data:** Organized in rows and columns with a fixed schema. Easily queryable using SQL. Example: Employee table with ID, Name, and Age.
* **Semi Structured Data:** Has structure but is not rigid. Uses key value pairs. Example: JSON and XML.
* **Unstructured Data:** No fixed format and difficult to query directly. Example: Images, videos, and text files.

## What Is a Database
A database is a structured storage system managed by a Database Management System (DBMS). It allows for efficient storage, fast retrieval, secure access, and data integrity. Examples include MySQL, PostgreSQL, and SQL Server.

## What Is SQL
SQL (Structured Query Language) is the standard language used to communicate with databases. It is used to define structure and to insert, update, delete, and retrieve data. SQL does not store data; it only interacts with the database server.

## What Is a Schema
A schema is a logical container inside a database that holds objects, similar to a folder.
### Types of Schema Objects
* **Tables:** Store actual data.
* **Views:** Virtual tables created from queries.
* **Stored Procedures:** Predefined reusable SQL logic.
* **Functions:** Return a single value.



## SQL Command Categories
SQL commands are grouped by their specific purpose:
* **DDL (Data Definition Language):** Defines or changes structure. Includes CREATE, ALTER, DROP, and TRUNCATE.
* **DML (Data Manipulation Language):** Works with the data inside tables. Includes INSERT, UPDATE, and DELETE.
* **DQL (Data Query Language):** Used to retrieve data. Primary command is SELECT.
* **DCL (Data Control Language):** Manages access control. Includes GRANT and REVOKE.
* **TCL (Transaction Control Language):** Manages transactions. Includes COMMIT, ROLLBACK, and SAVEPOINT.

## Data Integrity: NULL and Keys
* **NULL vs NOT NULL:** NULL represents missing or unknown values (not zero or empty). NOT NULL ensures a column must always contain a value to prevent invalid data.
* **Primary Key:** Uniquely identifies a row. It cannot be NULL and cannot have duplicates.
* **Foreign Key:** Links one table to another to ensure referential integrity.

## Table Relationships
* **Parent Child Relationship:** The Parent table stores master data (Primary Key), and the Child table depends on it (Foreign Key). This prevents orphan records.
* **Referential Actions:** * **RESTRICT:** Prevents deletion of a parent record if dependent child records exist.
    * **CASCADE:** Automatically deletes or updates child rows when the parent is changed.



## Constraints
Constraints enforce business rules at the database level to protect data quality.
* **PRIMARY KEY:** Unique identifier.
* **FOREIGN KEY:** Establishes relationships.
* **NOT NULL:** Mandatory value.
* **UNIQUE:** No duplicates allowed.
* **CHECK:** Condition based validation (e.g., Age must be 18 or older).
* **DEFAULT:** Provides an auto filled value when none is supplied.

## Database Inspection and Management
* **information_schema:** A system database that stores metadata. It allows you to inspect table structures and check constraints.
* **Safe Mode:** A MySQL safety feature that prevents accidental mass deletes. When ON, it requires a WHERE clause for updates or deletes.

## Removing Data: DROP vs DELETE vs TRUNCATE
* **DELETE:** Removes specific rows and can be rolled back.
* **TRUNCATE:** Removes all rows from a table. It is faster than DELETE but cannot be rolled back.
* **DROP:** Removes the entire table structure along with all its data.