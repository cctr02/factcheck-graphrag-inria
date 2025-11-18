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
gérer les dates différemment 
Etre sur que quand il preleve Donald Trump il preleve pas à part Donald et Trump 
Comment il détermine si ce Trump est une personne ou une organisation car ça apparait plusieurs fois 
chunkage propre 
répercuter dans le readme la normalisation des aretes MENTIONS CO Occurs Copresence 
Copresence il faudrait le weighted par la proba de présence du theme dans les chunsk considérés 
il sert à quoi le config.py ? j'ai l'impression qu'il est inutile si oui supprime le 
enlever le labelling par llm c'est le bottleneck trouver une stratégie rapide et efficace pour déterminer le theme à partir des keywords 
enlever les chuunks summaries par ollama, juste mettre le nom du chunk sur le noeud dans visualize et afficher le chunk entier quand on clique dessus 
Modifier les fonctions de statistiques pour qu'elles mettent les nouvelles fonctionnalités (copresence aussi) 
Afficher aussi les 10 keywords daans le summary de la création du graph 
entity faire attention il faut qu'aavec ou sans s ça soit la même entité (philippine et philippines) unifier les mots (nigeria nigerian
)
régler le problème pour les entités trouvées ( sri, lanka, sri lanka, sri lankan, lankan)