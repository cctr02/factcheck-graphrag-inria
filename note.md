Connexion 
ssh cedric.trinh@albatros.polytechnique.fr
ssh ctrinh@bb8.lix.polytechnique.fr

/mnt/disk5/prl_cedrictom/factcheckrag 
tmux
ctrl b then d
tmux attach -t 0


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

# Idées

Comment il détermine si ce Trump est une personne ou une organisation car ça apparait plusieurs fois 

entity retrieval par set in query is a subset of chunk 
clustering themes avec 10 min_topic size (using bertopic again) labeling by frequency based 
remove date node give higher score to chunk that has all entity 
tester la topic distribution au document lvl 

peut être essayer d'extract les entity au niveau document et les associer aux chunks ensuite pour une meilleure attribution entre ORG PERS etc 
relative date could be found via review date for entity node
faire des scores négatifs dans le sens où l'absence d'information est moins punitive (par exemple un chunk retrouvé mais avec des dates éloignées devraient être plus pénalisé qu'un chunk où il n'y a pas de dates )


- Construct graph for the entire corpus

- Share all the topics that are emerging in the chunks

- add edges between claim.
From claim found close to the query, do a relevance propagation to find other claims 

- try to run this : https://aclanthology.org/2025.findings-emnlp.852.pdf

