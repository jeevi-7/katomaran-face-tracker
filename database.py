
import os
import mysql.connector
import numpy as np


# --------------------------------------------------
# 1. DATABASE CONFIGURATION
# --------------------------------------------------

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": os.environ.get("KATOMARAN_DB_PASSWORD"),
    "database": "katomaran_db",
}


# --------------------------------------------------
# 2. DATABASE CONNECTION
# --------------------------------------------------

def get_connection():
    """Create and return a MySQL database connection."""

    if not DB_CONFIG["password"]:
        raise RuntimeError(
            "Set the KATOMARAN_DB_PASSWORD environment variable "
            "in your PowerShell terminal."
        )

    return mysql.connector.connect(**DB_CONFIG)


# --------------------------------------------------
# 3. SAVE A NEW VISITOR AND FACE EMBEDDING
# --------------------------------------------------

def save_visitor(visitor_id, embedding):
    """Save a visitor's 512-dimensional face embedding."""

    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.shape != (512,):
        raise ValueError(
            f"Expected a 512-dimensional embedding, got {embedding.shape}"
        )

    embedding_bytes = embedding.tobytes()

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO visitors (visitor_id, embedding)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE visitor_id = visitor_id
            """,
            (visitor_id, embedding_bytes),
        )

        connection.commit()

    finally:
        connection.close()


# --------------------------------------------------
# 4. LOAD SAVED VISITORS
# --------------------------------------------------

def load_visitors():
    """Load visitor IDs and their face embeddings from MySQL."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            "SELECT visitor_id, embedding FROM visitors"
        )

        visitors = {}

        for visitor_id, embedding_bytes in cursor.fetchall():
            embedding = np.frombuffer(
                embedding_bytes,
                dtype=np.float32,
            ).copy()

            visitors[visitor_id] = embedding

        return visitors

    finally:
        connection.close()


# --------------------------------------------------
# 5. COUNT UNIQUE REGISTERED VISITORS
# --------------------------------------------------

def get_unique_visitor_count():
    """Return the number of registered visitors."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM visitors")

        result = cursor.fetchone()

        return result[0]

    finally:
        connection.close()


# --------------------------------------------------
# 6. SAVE ENTRY OR EXIT EVENT
# --------------------------------------------------

def save_event(visitor_id, event_type, image_path):
    """
    Record a visitor entry or exit event.

    event_type must be 'entry' or 'exit'.
    image_path stores the path to the cropped face image.
    The database generates the event timestamp automatically.
    """

    if event_type not in ("entry", "exit"):
        raise ValueError(
            "event_type must be 'entry' or 'exit'"
        )

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO events (
                visitor_id,
                event_type,
                image_path
            )
            VALUES (%s, %s, %s)
            """,
            (visitor_id, event_type, image_path),
        )

        connection.commit()

    finally:
        connection.close()


# --------------------------------------------------
# 7. READ VISITOR EVENTS
# --------------------------------------------------

def get_visitor_events(visitor_id=None):
    """Return event records, optionally for one visitor."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        if visitor_id is None:
            cursor.execute(
                """
                SELECT event_id, visitor_id, event_type,
                       event_time, image_path
                FROM events
                ORDER BY event_time DESC
                """
            )
        else:
            cursor.execute(
                """
                SELECT event_id, visitor_id, event_type,
                       event_time, image_path
                FROM events
                WHERE visitor_id = %s
                ORDER BY event_time DESC
                """,
                (visitor_id,),
            )

        return cursor.fetchall()

    finally:
        connection.close()


# --------------------------------------------------
# 8. TEST DATABASE CONNECTION
# --------------------------------------------------

if __name__ == "__main__":
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("SELECT DATABASE()")
        print("Connected to:", cursor.fetchone()[0])

        print(
            "Unique visitors:",
            get_unique_visitor_count(),
        )

        cursor.execute("SELECT COUNT(*) FROM events")
        print("Total events:", cursor.fetchone()[0])

    finally:
        connection.close()
