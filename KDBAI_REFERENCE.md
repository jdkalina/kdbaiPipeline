# KDB.AI Client Reference Guide

This document provides a comprehensive reference for working with KDB.AI, a vector database optimized for AI applications, time series data, and semantic search.

## Table of Contents
- [Connection & Session](#connection--session)
- [Table Creation & Schema](#table-creation--schema)
- [Index Types](#index-types)
- [Data Operations](#data-operations)
- [Search Operations](#search-operations)
- [Hybrid Search](#hybrid-search)
- [Temporal Similarity Search (TSS)](#temporal-similarity-search-tss)
- [Metadata Filtering](#metadata-filtering)
- [RAG Integration](#rag-integration)
- [Embedding Models](#embedding-models)
- [Best Practices](#best-practices)

---

## Connection & Session

```python
import kdbai_client as kdbai
import os

# Create session (uses environment variable or default)
KDBAI_ENDPOINT = os.environ.get("KDBAI_ENDPOINT", "http://localhost:8082")
session = kdbai.Session(endpoint=KDBAI_ENDPOINT)

# Access a database
database = session.database("default")

# List all databases
session.databases()

# List tables in database
database.tables
```

---

## Table Creation & Schema

### Basic Table with Vector Index

```python
# Define schema - list of column definitions
schema = [
    {"name": "id", "type": "str"},
    {"name": "text", "type": "str"},
    {"name": "embeddings", "type": "float32s"}
]

# Define indexes - specify which columns to index and how
indexes = [
    {
        "name": "flat_index",
        "type": "flat",
        "column": "embeddings",
        "params": {"dims": 384, "metric": "CS"}
    }
]

# Create table
table = database.create_table(
    "my_table",
    schema=schema,
    indexes=indexes
)
```

### Supported Column Types
- `str` - String/text data
- `int64` - 64-bit integers
- `float32` - 32-bit floats
- `float64` - 64-bit floats
- `float32s` - List of 32-bit floats (for embeddings)
- `bytes` - Binary data
- `datetime64[ns]` - Timestamps

### Table with Multiple Indexes (Hybrid Search)

```python
schema = [
    {"name": "id", "type": "str"},
    {"name": "text", "type": "str"},
    {"name": "dense_embeddings", "type": "float32s"},
    {"name": "sparse_embeddings", "type": "float32s"}
]

indexes = [
    {
        "name": "dense_index",
        "type": "flat",
        "column": "dense_embeddings",
        "params": {"dims": 384, "metric": "CS"}
    },
    {
        "name": "sparse_index",
        "type": "bm25",
        "column": "sparse_embeddings",
        "params": {"k": 1.25, "b": 0.75}
    }
]

table = database.create_table("hybrid_table", schema=schema, indexes=indexes)
```

---

## Index Types

| Index Type | Description | Best For |
|------------|-------------|----------|
| `flat` | Exact nearest neighbor search (in-memory) | Small datasets, exact results |
| `qFlat` | On-disk flat index | Large datasets, memory-efficient |
| `hnsw` | Hierarchical Navigable Small World | Fast approximate search |
| `qHnsw` | On-disk HNSW | Large datasets with speed needs |
| `ivf` | Inverted File index | Clustered data |
| `ivfpq` | IVF with Product Quantization | Very large datasets |
| `bm25` | Sparse keyword index | Text keyword search |
| `tss` | Temporal Similarity Search | Time series patterns |

### Distance Metrics
- `L2` - Euclidean distance (smaller = more similar)
- `CS` - Cosine similarity (larger = more similar)
- `IP` - Inner product

### Index Configuration Examples

```python
# Flat index (exact search)
{"name": "flat", "type": "flat", "column": "embeddings",
 "params": {"dims": 384, "metric": "CS"}}

# HNSW index (approximate, fast)
{"name": "hnsw", "type": "hnsw", "column": "embeddings",
 "params": {"dims": 384, "metric": "L2", "efConstruction": 200, "M": 16}}

# On-disk flat index
{"name": "qflat", "type": "qFlat", "column": "embeddings",
 "params": {"dims": 1024, "metric": "CS"}}

# BM25 sparse index
{"name": "sparse", "type": "bm25", "column": "sparse_vectors",
 "params": {"k": 1.25, "b": 0.75}}
```

---

## Data Operations

### Inserting Data

```python
import pandas as pd

# Create DataFrame matching schema
df = pd.DataFrame({
    "id": ["doc1", "doc2", "doc3"],
    "text": ["Hello world", "KDB.AI is great", "Vector search"],
    "embeddings": [embedding1, embedding2, embedding3]  # List of float lists
})

# Insert data
result = table.insert(df)
# Returns: {'rowsInserted': 3}

# Batch insertion for large datasets
batch_size = 2000
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    table.insert(batch)
```

### Querying Data

```python
# Retrieve all data
all_data = table.query()

# Query with filter
filtered = table.query(filter=[("like", "text", "hello*")])

# Query specific columns
table.query(columns=["id", "text"])
```

### Table Management

```python
# Check table indexes
table.indexes

# Drop table
table.drop()

# Drop database
database.drop()
```

---

## Search Operations

### Basic Similarity Search

```python
# Search with a single query vector
results = table.search(
    vectors={"flat_index": [query_embedding]},
    n=5  # Return top 5 results
)

# Results is a list of DataFrames (one per query)
# Each DataFrame includes __nn_distance column
for result_df in results:
    print(result_df)
```

### Multi-Query Search

```python
# Search with multiple queries at once
query_vectors = [query1, query2, query3]
results = table.search(
    vectors={"flat_index": query_vectors},
    n=10
)
# Returns list of 3 DataFrames
```

### Search with Sorting

```python
results = table.search(
    vectors={"flat_index": [query_embedding]},
    n=10,
    sort_columns=["timestamp"]  # Sort by additional column
)
```

### Search with Aggregation

```python
results = table.search(
    vectors={"flat_index": [query_embedding]},
    n=10,
    aggs={"avg_score": ["avg", "score"]},
    group_by=["category"]
)
```

---

## Hybrid Search

Combine dense (semantic) and sparse (keyword) search for better results.

```python
# Generate both embedding types
dense_query = embedding_model.embed(query_text)
sparse_query = bm25_tokenize(query_text)

# Hybrid search with weighted combination
results = table.search(
    vectors={
        "dense_index": [dense_query],
        "sparse_index": [sparse_query]
    },
    index_params={
        "dense_index": {"weight": 0.5},
        "sparse_index": {"weight": 0.5}
    },
    n=10
)
```

---

## Temporal Similarity Search (TSS)

Search for similar patterns in time series data without embedding transformation.

### Non-Transformed TSS Setup

```python
schema = [
    {"name": "id", "type": "str"},
    {"name": "timestamp", "type": "datetime64[ns]"},
    {"name": "price", "type": "float32s"}
]

indexes = [
    {
        "name": "tss_index",
        "type": "tss",
        "column": "price",
        "params": {"dims": 100}  # Window size
    }
]

table = database.create_table("timeseries", schema=schema, indexes=indexes)
```

### TSS Search Operations

```python
# Find similar time series patterns
results = table.search(
    vectors={"price": [time_series_window]},
    n=10,
    type="tss"
)

# Find outliers (most dissimilar)
outliers = table.search(
    vectors={"price": [query_window]},
    n=-10,  # Negative n for outliers
    type="tss"
)

# Dynamic query sizes in single search
results = table.search(
    vectors={"price": [full_vector, half_vector, quarter_vector]},
    n=10,
    type="tss"
)
```

---

## Metadata Filtering

Filter search results by metadata columns.

### Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `=` | Equals | `("=", "category", "tech")` |
| `<>` | Not equals | `("<>", "status", "deleted")` |
| `<` | Less than | `("<", "price", 100)` |
| `>` | Greater than | `(">", "score", 0.5)` |
| `<=` | Less or equal | `("<=", "count", 10)` |
| `>=` | Greater or equal | `(">=", "rating", 4)` |
| `within` | Range (inclusive) | `("within", "year", (2020, 2024))` |
| `like` | Pattern match | `("like", "name", "test*")` |

### Filtered Search

```python
# Search with single filter
results = table.search(
    vectors={"flat_index": [query_embedding]},
    n=5,
    filter=[("=", "category", "technology")]
)

# Search with multiple filters (AND)
results = table.search(
    vectors={"flat_index": [query_embedding]},
    n=5,
    filter=[
        ("=", "category", "technology"),
        ("within", "year", (2020, 2024)),
        (">", "score", 0.8)
    ]
)
```

### Fuzzy Filtering

```python
# Pattern matching with wildcards
results = table.search(
    vectors={"flat_index": [query_embedding]},
    n=5,
    filter=[("like", "title", "*python*")]
)
```

---

## RAG Integration

### LangChain Integration

```python
from langchain_community.vectorstores import KDBAI
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# Create vector store
vecdb = KDBAI(
    table=table,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)

# Add documents
vecdb.add_documents(chunks)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=vecdb.as_retriever(
        search_kwargs={"k": 10, "index": "flat_index"}
    ),
    return_source_documents=True
)

# Query
result = qa_chain.invoke({"query": "What is KDB.AI?"})
print(result["result"])
```

### LlamaIndex Integration

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.kdbai import KDBAIVectorStore

# Create vector store
vector_store = KDBAIVectorStore(table=table)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index from documents
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is KDB.AI?")
```

### LlamaIndex Hybrid Search

```python
from llama_index.vector_stores.kdbai import KDBAIVectorStore

# Enable hybrid search
vector_store = KDBAIVectorStore(table=table, hybrid_search=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

# Create hybrid retriever
retriever = index.as_retriever(
    vector_store_query_mode="hybrid",
    similarity_top_k=5,
    vector_store_kwargs={
        "index": "dense_index",
        "indexWeight": 0.5,
        "indexSparse": "sparse_index",
        "indexSparseWeight": 0.5
    }
)
```

---

## Embedding Models

### Common Models Used in Samples

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `BAAI/bge-small-en-v1.5` (FastEmbed) | 384 | General text, lightweight |
| `all-MiniLM-L6-v2` | 384 | General text |
| `all-mpnet-base-v2` | 768 | Higher quality text |
| `text-embedding-3-small` (OpenAI) | 1536 | Production text |
| `voyage-multimodal-3` (Voyage AI) | 1024 | Text + images |
| ResNet-50 features | 2048 | Images only |

### FastEmbed Example

```python
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = list(model.embed(["Hello world", "KDB.AI rocks"]))
```

### Sentence Transformers Example

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["Hello world", "KDB.AI rocks"])
```

### OpenAI Embeddings

```python
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Hello world"]
)
embedding = response.data[0].embedding
```

---

## Best Practices

### Performance Optimization

1. **Choose the right index type:**
   - `flat` for small datasets (<100K vectors) requiring exact results
   - `hnsw` for larger datasets with speed requirements
   - `qFlat`/`qHnsw` for very large datasets (memory-efficient)

2. **Batch operations:**
   ```python
   batch_size = 2000
   for i in range(0, len(data), batch_size):
       table.insert(data.iloc[i:i+batch_size])
   ```

3. **Use metadata filtering** to reduce search space before vector comparison

4. **Hybrid search** combines keyword and semantic search for better results

### Common Configuration Values

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Chunk size | 500-1000 chars | For text splitting |
| Chunk overlap | 50-100 chars | Context preservation |
| Top-k results | 5-10 | Balance recall/precision |
| Batch size | 2000-10000 | Memory management |
| BM25 k | 1.25 | Standard value |
| BM25 b | 0.75 | Standard value |

### Error Handling

```python
try:
    table = database.create_table("my_table", schema=schema, indexes=indexes)
except Exception as e:
    if "already exists" in str(e):
        table = database.table("my_table")
    else:
        raise
```

### Cleanup Pattern

```python
# Clean up existing table before recreation
table_name = "my_table"
if table_name in [t.name for t in database.tables]:
    database.table(table_name).drop()

table = database.create_table(table_name, schema=schema, indexes=indexes)
```

---

## Sample Notebooks Overview

The repository contains 33 Jupyter notebooks organized by use case:

| Category | Notebooks | Description |
|----------|-----------|-------------|
| Quickstarts | 1 | Basic Python quickstart |
| Course Materials | 3 | Educational KDB.AI course |
| Hybrid Search | 1 | Dense + sparse combination |
| Metadata Filtering | 2 | Filtering search results |
| Document Search | 2+ | PDF semantic search |
| Index Comparisons | 2 | qFlat vs qHnsw |
| Time Series | 5 | TSS transformed/non-transformed |
| Pattern Matching | 1 | Sensor data patterns |
| Image Search | 1 | MRI brain scans |
| Multimodal Search | 3 | Text + image retrieval |
| RAG | 6+ | Various RAG implementations |
| LlamaIndex | 4 | Framework integration |
| Video RAG | 2 | Video content retrieval |

---

## Quick Reference

```python
import kdbai_client as kdbai
import pandas as pd

# Connect
session = kdbai.Session(endpoint="http://localhost:8082")
db = session.database("default")

# Create table
schema = [
    {"name": "id", "type": "str"},
    {"name": "text", "type": "str"},
    {"name": "embeddings", "type": "float32s"}
]
indexes = [{"name": "idx", "type": "flat", "column": "embeddings",
            "params": {"dims": 384, "metric": "CS"}}]
table = db.create_table("docs", schema=schema, indexes=indexes)

# Insert
df = pd.DataFrame({"id": ["1"], "text": ["hello"], "embeddings": [[0.1]*384]})
table.insert(df)

# Search
results = table.search(vectors={"idx": [[0.1]*384]}, n=5)

# Filter search
results = table.search(vectors={"idx": [[0.1]*384]}, n=5,
                       filter=[("like", "text", "hel*")])

# Cleanup
table.drop()
```
