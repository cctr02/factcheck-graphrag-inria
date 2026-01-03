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


"{""text"": ""It is taken from a campaign event in South Carolina on February 27, 2020, and the full exchange can be seen from the 10:50 minute mark of this video:\n\n“By the way how many of you did really well with that $1.9 trillion tax cut… ?” Biden asks, referring to major 2017 tax reforms enacted under Trump. Biden’s position is that the middle class was largely left out of those changes, and in the video he addresses someone who acknowledged benefiting from Trump’s cuts.\n“Well you did? Well that’s great."", ""type"": ""chunk"", ""doc_id"": ""33"", ""chunk_idx"": 3}"

"{""text"": ""Citizenship and Immigration Service (USCIS). Through this program, all of our new employees are screened by the government to help verify their legal immigration status.” \n Current work at the border\nThroughout his campaign for the presidency, Donald Trump promised to build a wall along the southern US border. On January 25, 2017, five days after taking office, Trump issued this executive order directing the Department of Homeland Security to identify and allocate funding for such a barrier."", ""type"": ""chunk"", ""doc_id"": ""4"", ""chunk_idx"": 6}"




entity retrieval par set in query is a subset of chunk 
clustering themes avec 10 min_topic size (using bertopic again) labeling by frequency based 
remove date node give higher score to chunk that has all entity 
tester la topic distribution au document lvl 
logger aussi le nombre de chunk dans topic ID - 1 pour voir combien de chunk sont outliers 
peut être essayer d'extract les entity au niveau document et les associer aux chunks ensuite pour une meilleure attribution entre ORG PERS etc 
relative date could be found via review date for entity node
faire des scores négatifs dans le sens où l'absence d'information est moins punitive (par exemple un chunk retrouvé mais avec des dates éloignées devraient être plus pénalisé qu'un chunk où il n'y a pas de dates )