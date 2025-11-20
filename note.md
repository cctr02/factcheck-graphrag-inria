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

# Idées
Proper sentence chunking 

lemmatize keywords given by BERTopic and keep labeling it 

gérer les dates différemment group it by year 

stoquer les embeddings dans un vector database 

Etre sur que quand il preleve Donald Trump il preleve pas à part Donald et Trump 

Comment il détermine si ce Trump est une personne ou une organisation car ça apparait plusieurs fois 
chunkage propre 

Afficher aussi les 10 keywords daans le summary de la création du graph 

est-ce que inverse frequency factor c'est standard ? si oui mettre un lien 
mettre que le relevance propagation s'arrête quand ça atteint un threshold 
