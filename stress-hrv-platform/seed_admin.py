import sqlite3
import hashlib
import os
import sys

# Update this path if needed, but based on workspace info it should be correct
DB_PATH = r"D:\Olivia\stress-hrv-platform\fog_orchestrator\data\federated_local.db"

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        120_000,
    ).hex()

def seed_admin():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        sys.exit(1)

    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Determine table structure to be safe
    cursor.execute("PRAGMA table_info(users)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if 'email' not in columns:
        print("Error: 'users' table does not have 'email' column or table does not exist.")
        conn.close()
        return

    email = "admin@example.com"
    password_text = "admin123"

    print(f"Checking for user: {email}")
    cursor.execute("SELECT id, name FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()

    if row:
        print(f"User '{email}' already exists (ID: {row[0]}, Name: {row[1]}).")
        # Optional: Reset password if asked, but for now just report existence.
    else:
        print(f"User '{email}' not found. Creating...")
        salt = os.urandom(16).hex()
        p_hash = _hash_password(password_text, salt)
        
        try:
            cursor.execute("""
              INSERT INTO users (name, email, password_hash, password_salt, active)
              VALUES (?, ?, ?, ?, 1)
            """, ("Admin User", email, p_hash, salt))
            conn.commit()
            print(f"Successfully created user '{email}' with password '{password_text}'.")
        except Exception as e:
            print(f"Error creating user: {e}")

    conn.close()

if __name__ == "__main__":
    seed_admin()
