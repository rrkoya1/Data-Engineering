# Session 2 — Big Data, Hadoop and Spark

## Big Data: Why Traditional Systems Fail
Big Data refers to data that is too large, too fast, or too complex for traditional databases and single machines to handle efficiently. 

As data grew into terabytes and petabytes, systems faced:
* Storage limits on a single server
* Long processing times
* Frequent hardware failures

This created the need for distributed systems, where data and computation are spread across multiple machines.

## Why Hadoop Was Introduced
Hadoop uses many inexpensive machines working together as a cluster to solve three problems:
1. Storing huge data reliably
2. Processing data in parallel
3. Handling failures automatically

## HDFS — Hadoop Distributed File System
HDFS is the storage layer. Large files are split into blocks and distributed across machines. Each block is replicated (usually 3 copies) so data is not lost if a machine fails.

### HDFS Components
* **NameNode:** Stores metadata only (file names and block locations). It does not store actual data.
* **DataNode:** Stores the actual data blocks and handles read/write operations.

## YARN Architecture — Resource Management
YARN (Yet Another Resource Negotiator) acts as the operating system of the cluster.
* **Resource Manager:** The global master that decides how resources are distributed.
* **Node Manager:** Runs on each node to execute assigned tasks.

## MapReduce — Hadoop’s Processing Model
MapReduce is a programming paradigm for batch processing.
1. **Map Phase:** Breaks input data into key–value pairs.
2. **Reduce Phase:** Aggregates values by key.

**Clarification:** MapReduce is a computation model, while the Silver layer is a data-quality layer. MapReduce does not enforce quality; it only processes data.

## Why Hadoop MapReduce Is Slow
It is disk-based. It reads from and writes to the disk for every stage, causing heavy disk I/O and long execution times. This led to the development of Spark.

## Apache Spark — Unified In-Memory Computation
Spark processes data in RAM instead of writing everything to disk. This makes it much faster and ideal for analytics and machine learning. It can still use HDFS, S3, or Cloud storage for persistence.

### Spark Architecture
* **Spark Application:** The job submitted by the user.
* **Driver Program:** The brain that creates the execution plan (DAG) and coordinates tasks.
* **Cluster Manager:** Allocates resources (YARN, Kubernetes, or Standalone).
* **Executors:** Worker nodes that execute tasks and cache data in memory.

## RDD — Resilient Distributed Dataset
RDDs are Spark’s core abstraction. They are distributed, immutable, and fault-tolerant. They allow for in-memory computation and automatic recomputation if a failure occurs.

## DAG — Directed Acyclic Graph
Spark builds a DAG to represent the logical execution flow. It optimizes the execution and runs only when an action is triggered.

## Hadoop vs Spark Comparison

| Feature | Hadoop (MapReduce) | Spark |
| :--- | :--- | :--- |
| Processing | Disk-based | In-memory |
| Speed | Slow | Fast |
| Workloads | Batch only | Batch, Streaming, ML |
| Ease of use | Complex | Developer-friendly |

### Component Mapping
* **Hadoop NameNode** = Spark Driver
* **Hadoop DataNode** = Spark Executor
* **Hadoop MapReduce** = Spark DAG
* **Hadoop Disk I/O** = Spark RAM

## Big Data Ecosystem and Cloud Platforms
Concepts remain the same across cloud providers:
* **AWS:** S3 (Lake), Glue (ETL), Redshift (Warehouse)
* **Azure:** Blob Storage, Data Factory, Synapse
* **GCP:** Cloud Storage, Dataflow, BigQuery

## Data Governance and Quality
* **Data Lineage:** Tracks data origin and changes.
* **Data Cataloging:** Searchable inventory (AWS Glue Catalog, Apache Atlas).
* **Data Quality:** Validation via schema checks, null checks, and tools like Great Expectations.
* **Unit Testing:** Tests ETL logic to prevent failures.

## Data Delivery and Real-Time Systems
* **Delivery:** Data is exposed via Dashboards (Power BI, Tableau) or APIs (FastAPI, Flask).
* **Real-Time Tools:** Kafka, Spark Streaming, and Kinesis are used for fraud detection and live alerts.