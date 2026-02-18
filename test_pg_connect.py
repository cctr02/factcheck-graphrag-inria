import psycopg2

conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    dbname="knowledge_graph",
    user="postgres",
    password="postgres"
)

cur = conn.cursor()
cur.execute("SELECT current_database(), current_user;")
print(cur.fetchone())

cur.close()
conn.close()

print("✅ CONNECTED TO knowledge_graph")
