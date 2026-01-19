  **Session 1** 

&nbsp;

&nbsp;                                                                                                                               Ritesh Reddy Koya 



**Data Engineering:** 



Raw data from logs, transactions, and APIs is messy, incomplete, and untrusted. 



Data Engineering exists to build the bridge from raw data to usable insight. 



**Data (Raw):** Uninterpreted facts (e.g., 100, "John") 



&nbsp;



**Information (Meaning):** Processed data (e.g., “John scored 100 marks”) 



Meaning is created, not stored. 



Data cannot express itself, Data Engineers make data usable, while Analysts and Data Scientists give it a voice. 



**The Data Engineering Lifecycle:** 



This is the non-optional journey data always takes. 



If any stage fails, the entire pipeline fails. 



**Collect:** Data is born in apps, APIs, sensors. It is raw and untrusted. 



**Ingest:** Moving data into the system 



**Batch:** scheduled, cheaper  



**Real-time:** continuous, complex 



**Processing (Transformation):** Where the mess becomes clean 



Cleaning, deduplication, joins, aggregations 



SQL and Spark live here 



**Store:** 



Data Lakes: raw, flexible 



Data Warehouses: structured, fast 



**Analyze:** Clean data is now queryable (BI, analytics). 



**Consume:** dashboards, APIs, ML models. 



&nbsp;



**ETL vs. ELT:** 



**ETL:** 



Extract -> Transform -> Load 



Data is cleaned before storage. 



Good for strict control, but less flexible. 



**ELT:** 



Extract -> Load -> Transform 



Raw data is stored first, then transformed later. 



Why ELT? 



Cloud storage is cheap, and keeping raw data allows flexible reprocessing later. 



&nbsp;



**The Tech Stack: Hadoop, Spark, Snowflake:** 



**Hadoop:** 



The foundation. Introduced distributed storage (HDFS) and cluster-based processing. 



**Spark:** 



The modern processing engine. 



Uses parallel processing and in-memory computation, making it much faster than older systems. 



**Snowflake:** 



A cloud-native data warehouse that separates compute (work) from storage (data). 



Scales automatically and is built for analytics, not daily transactions. 

&nbsp;

**Automation vs. Orchestration:** 



**Automation:** 



A single task runs by itself. 



**Orchestration:** 



Managing the big picture, task order, dependencies, retries, and failures. 



Example: Don’t start Task B until Task A finishes successfully. 



&nbsp;Airflow is the most commonly used orchestration tool. 



&nbsp;



**Architecture diagram:** 



The diagram illustrates a data engineering system that takes raw data inputs from different sources i.e. databases, APIs, files and event streams enabling it to be accessed by consumers (dashboards and ML models) throughout the whole life cycle of raw data starting from Generation by Operational Systems (OS) through ingestion into a Centralized Storage Layer (CSL) at which point Veracity does not exist (raw data) until through Distributed Processing Engines (DPEs) the data is cleansed, joined and transformed (to create Structure) so the cleansed/transformed data stored in Enterprise Data Warehouses (EDWs) (the recipient of data from all sources). There are many Business Intelligence (BI) tools/reports/APIs/ML Models being utilised to consume data received from Enterprise Data Warehouses, whilst Orchestration Tools are lightly integrated into the whole flow allowing the management of data, monitoring and scheduling of dependencies between steps in the process, showing the Lifecycle (complete process) of Data, not just SQL or Databases, etc. 



**Data Sources (The Input)**: Raw and untrusted data originates from Operational databases (MySQL/Postgres), APIs, Files (JSON/CSV), and Event streams (Kafka). 



Ingestion: The process of reliably moving data into the platform via Batch (scheduled) or Real-time (streaming) using Kafka or Python. 



**Data Lake (Raw Storage):** A cheap, scalable "Single Source of Truth" (S3/GCS) that stores data in its original format using schema-on-read. 



**Processing \& Refined Layers:** The "Cleaning" phase. Spark or SQL engines use parallel processing to deduplicate and join data, turning it into standardized, business-ready datasets. 



**Data Warehouse (Analytics):** Structured storage (Snowflake/BigQuery) optimized for querying. It supports BI and reporting, not transactional workloads. 



**Consumption:** The final output where business value is created via Dashboards (Tableau/Power BI), APIs, and ML models. 



**Orchestration:** The "Brain" (Airflow/Prefect) that manages dependencies, schedules jobs, and handles failures across the entire pipeline. 



Here is a concise, paragraph-based breakdown of the Medallion Architecture, keeping all your technical tiers intact. 



&nbsp; 



**Bronze Layer:** The Raw Historical Record 



The Bronze layer is your landing zone. It stores data exactly as it arrives from sources like APIs and event streams, prioritizing data availability and traceability over usability. Because it uses schema-on-read, it contains uncleaned, unvalidated data, including duplicates or errors. Its primary job is to act as a historical record and recovery point if you ever need to reprocess the data. 



&nbsp; 



**Silver Layer:** The Trusted Foundation 



The Silver layer is where the "mess" gets organized. Here, data from the Bronze layer undergoes deduplication, filtering, and basic validation. In this phase, data structures are enforced, multiple sources are joined, and initial business logic begins to appear. This layer provides high-quality, cleansed and refined data that is reliable enough for analysts to explore. 



&nbsp; 



**Gold Layer:** The Value Creator 



The Gold layer is the final destination, containing fully curated, analytics-ready datasets. Data here is transformed into aggregations and business metrics, often modeled into facts and dimensions for maximum performance. This is the business-ready layer consumed by dashboards, reports, and ML models to create actual impact. 

&nbsp;

