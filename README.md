# Setup

1. Install dependencies.

```sh
uv sync
```

2. Run streamlit server.

```sh
uv run streamlit run streamlit_frontend.py
```

3.  Postgres setup

```sh
# Run postgres
docker compose up

# Connect to postgres
psql -h localhost -p 5434 -U postgres -d vectordb

# Create PG Vector extension
CREATE EXTENSION IF NOT EXISTS vector

# Create Tables
CREATE TABLE faqs (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    embedding VECTOR(1024)
);

CREATE TABLE inventory (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price NUMERIC NOT NULL,
    type TEXT NOT NULL,
    description TEXT NOT NULL,
    embedding VECTOR(1024)
);

# Create Indexes
CREATE INDEX faqs_embedding_idx
ON faqs
USING hnsw (embedding vector_cosine_ops)

CREATE INDEX inventory_embedding_idx
ON inventory
USING hnsw (embedding vector_cosine_ops);
```

# Running the application

1. Load data to database. This step creates embeddings for faqs and inventory and stores the data to database. This step should only be done once. In `vector_store.py` file, uncomment the code under `if __name__ == "__main__"` and run

```sh
uv run vector_store.py
```

2. Run streamlit server.

```sh
uv run streamlit run server.py
```
