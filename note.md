Connexion 
ssh cedric.trinh@albatros.polytechnique.fr
ssh ctrinh@bb8.lix.polytechnique.fr


/mnt/disk4/younes/fcbureau_data

articles.csv       claims.csv              embeddings_french.pt       english_corpus.json       french_corpus.json
bm25_english.pkl   embeddings_english.npy  embeddings.ipynb           english_fact_checks.csv   french_fact_checks.csv
bm25_french.pkl    embeddings_english.pt   english_claim_reviews.csv  french_claim_reviews.csv
claim_reviews.csv  embeddings_french.npy   english_claims.csv         french_claims.csv

Voir les headers : 
python3 - <<'EOF'
import csv
with open("french_fact_checks.csv", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        print(row)
        if i == 4:
            break
EOF

['id', 'url', 'title', 'body', 'language', 'publisher_name', 'publisher_site', 'review_date']

# Pour lancer factcheckbureau
Lancer Ubuntu
cd /mnt/c/Users/cedri/PythonProject/PRL/factcheckbureau
bash scripts/run.sh

# Spacy download
uv run python -m spacy download en_core_web_sm