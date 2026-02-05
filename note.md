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

fais des edges between claim 
et depuis les claim similaires à la query refaire un relevance propagation pour retrouver d'autre claims 

(https://aclanthology.org/2025.findings-emnlp.852.pdf):
regarder si ça ça run bien 

- Construct graph for the entire corpus

- Share all the topics that are emerging in the chunks

- add edges between claim.
From claim found close to the query, do a relevance propagation to find other claims 

- try to run this : https://aclanthology.org/2025.findings-emnlp.852.pdf


- 5th strategy
From claim close to query, compute the distance between distribution of topics between chunks found through graph traversal from the claim and the query to give a score to the final chunk. Give more weight to chunk that has topic that are highly relevant regarding TF IDF (regarding the confidence that a topic is present in a chunk) 
6th

- 6th strategy 
from the claim also
From the query we extract keyBERT phrase and we then compute semantic similarity with the chunk found through claim retrieval and we add score based on this 

il y a un probleme dans le logging pour build graph
Filtered language=='english': kept 38087/829687 rows
il y a au maximum 38087 documents quoi qu'il arrive c'est juste qu'une ligne n'est pas forcément un seul doc, j'pense que dans le code c'est bien remarqué mais pas dans le logging vérifie que c'est bien fait dans les 2 (regarde la structure du csv pour être sûr)
 

 garde seulement la barre tqdm et enleve le log qui dit completed x / num epoch dans le logging quand tu build le graph
 2026-02-05 22:45:45,037 - BERTopic - Embedding - Transforming documents to embeddings.
Batches: 100%|█████████████████████████████████████████████████████████████████| 13/13 [00:03<00:00,  3.29it/s]
2026-02-05 22:45:49,012 - BERTopic - Embedding - Completed ✓
2026-02-05 22:45:49,012 - BERTopic - Dimensionality - Fitting the dimensionality reduction algorithm
UMAP(angular_rp_forest=True, metric='cosine', min_dist=0.0, n_components=5, verbose=True)
Thu Feb  5 22:45:49 2026 Construct fuzzy simplicial set
Thu Feb  5 22:45:49 2026 Finding Nearest Neighbors
Thu Feb  5 22:45:58 2026 Finished Nearest Neighbor Search
Thu Feb  5 22:46:04 2026 Construct embedding
Epochs completed:   0%|                                                                           0/500 [00:00]completed  0  /  500 epochs
Epochs completed:   0%| ▏                                                                         1/500 [00:03]completed  50  /  500 epochs
        completed  100  /  500 epochs
        completed  150  /  500 epochs
        completed  200  /  500 epochs
        completed  250  /  500 epochs
        completed  300  /  500 epochs
Epochs completed:  63%| ████████████████████████████████████████████▌                           314/500 [00:03]completed  350  /  500 epochs
        completed  400  /  500 epochs
        completed  450  /  500 epochs
Epochs completed: 100%| ███████████████████████████████████████████████████████████████████████ 500/500 [00:03]

pourquoi il y a un double logging lors du process des kewords :
Step 3: Adding themes and HAS_TOPIC edges...
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 8 (removed 2 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 7 (removed 3 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 8 (removed 2 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 10 (removed 0 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 10 (removed 0 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 8 (removed 2 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 9 (removed 1 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 10 (removed 0 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 10 (removed 0 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 8 (removed 2 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 10 (removed 0 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 7 (removed 3 redundant)
22:46:08 | build_graph          | INFO | Labeling 12 topics using batched LLM requests...
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 8 (removed 2 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 7 (removed 3 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 8 (removed 2 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 10 (removed 0 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 10 (removed 0 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 8 (removed 2 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 9 (removed 1 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 10 (removed 0 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 10 (removed 0 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 8 (removed 2 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 10 (removed 0 redundant)
22:46:08 | build_graph          | DEBUG | Processed keywords: 10 -> 7 (removed 3 redundant)