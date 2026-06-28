import json
import os

import psycopg
import streamlit as st
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

# Create a psycopg2 postgres connection
conn = psycopg.connect(
    host="localhost",
    port=5434,
    dbname="vectordb",
    user="postgres",
    password="postgres",
)

# Register pgvector python package to this connection. This package supports the type converstion (serialization and deserialization) between numpy arrays and postgres native vector type.
register_vector(conn)


# Load model. Downloads automatically if not present
@st.cache_resource
def load_model():
    return SentenceTransformer(os.environ["EMBEDDINGS_MODEL"], device="mps")


def load_faqs():
    # Load faqs from file
    with open("./datasource/FAQ.json", "r") as file:
        faqs = json.load(file)

    # Create documents from faqs
    documents = [
        f"Question: {faq['question']}\n\nAnswer: {faq['answer']}" for faq in faqs
    ]

    # Create embeddings from documents
    model = load_model()
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        batch_size=64,
    )

    # Insert data to db
    rows = [
        (
            faq["question"],
            faq["answer"],
            embedding,
        )
        for faq, embedding in zip(faqs, embeddings)
    ]
    with conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO faqs (question, answer, embedding)
                VALUES (%s, %s, %s)
                """,
                rows,
            )


def load_inventory():
    # Load inventory from file
    with open("./datasource/inventory.json", "r") as file:
        inventory = json.load(file)

    # Create documents from inventory
    documents = [
        f"Product: {item['name']}\n\nType: {item['type']}\n\nDescription: {item['description']}"
        for item in inventory
    ]

    # Create embeddings from documents
    model = load_model()
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        batch_size=64,
    )

    # Insert data to db
    rows = [
        (
            item["id"],
            item["name"],
            item["quantity"],
            item["price"],
            item["type"],
            item["description"],
            embedding,
        )
        for item, embedding in zip(inventory, embeddings)
    ]
    with conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO inventory (id, name, quantity, price, type, description, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )


def search_faqs(query: str):
    model = load_model()
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                question,
                answer
            FROM faqs
            ORDER BY embedding <=> %s
            LIMIT 3;
            """,
            (query_embedding,),
        )

        return [
            {"question": question, "answer": answer}
            for question, answer in cur.fetchall()
        ]


def search_inventory(query: str):
    model = load_model()
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                name,
                price,
                quantity,
                type,
                description
            FROM inventory
            ORDER BY embedding <=> %s
            LIMIT 5;
            """,
            (query_embedding,),
        )

        return [
            {
                "id": id,
                "name": name,
                "price": float(price),
                "quantity": quantity,
                "type": type,
                "description": description,
            }
            for id, name, price, quantity, type, description in cur.fetchall()
        ]


if __name__ == "__main__":
    # Only run this once.
    # load_faqs()
    # load_inventory()
    pass
