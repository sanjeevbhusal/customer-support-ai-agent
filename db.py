"""The shared Postgres connection pool used across the app (auth, tools,
tracing, vector search)."""

from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

load_dotenv()


def _configure(conn):
    # Register pgvector on each connection so numpy arrays serialize to and from
    # the postgres vector type (needed by the embedding-search queries).
    register_vector(conn)


pool = ConnectionPool(
    "host=localhost port=5434 dbname=vectordb user=postgres password=postgres",
    kwargs={"autocommit": True},
    configure=_configure,
    min_size=1,
    max_size=10,
    open=True,
)
