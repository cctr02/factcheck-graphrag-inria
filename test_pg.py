import psycopg2

conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="postgres"
)

conn.autocommit = True
cur = conn.cursor()

cur.execute("SELECT datname FROM pg_database;")
dbs = [r[0] for r in cur.fetchall()]
print("DBs:", dbs)

if "knowledge_graph" not in dbs:
    cur.execute("CREATE DATABASE knowledge_graph;")
    print("Created knowledge_graph")
else:
    print("knowledge_graph already exists")

cur.close()
conn.close()

print("✅ Done")
