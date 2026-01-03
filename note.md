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

####

Pour la mise en page fais une vraie mise en page comme un article scientifique 
donc réfère toi sur ce qui est fait en terme de police, de taille, de marge, de présentation du titre, des titres des parties de numérotation des pages le fait que ça soit en double colonne 
Je te fais une liste des choses à changer dans le rapport : 
Introduction 
Explique ce qu'est le fact checking et en quoi c'est relevant mets note pour rediriger vers l'annexe où tu mets un exemple entier d'un article de factcheck tel qu'on le stock dans notre database (tu peux en voir un exemple dans data) donne les attributs de ce qu'on stock. Fais un état sur les caractéristiques des données qu'on a et dis qu'on les a obtenu grace au lab Inria 
Ensuite décris ce qu'on veut obtenir donc un systeme permettant de retrouver des articles en fonction d'une claim donc détaille ce qu'est une claim et ce qu'on désire en résultat donc les articles par exemple les plus pertinents. décris les use case qu'on veut en fait, et dis que ça pourrait s'étendre à d'autres domaines mais qu'on prend le cas spécifique du factchecking 
Ensuite parle de l'état de l'art sur ce qui existe comme approche pour résoudre ce probleme 
Enfin parle des spécificités de notre projet donc qu'on fait un knowledgre grpah avec 4 types de nodes, avec toutes les arêtes qu'on utilise  et dis qu'on va détailler la suite. introduis en même temps la figure qui présente le schéma de notre graph 
puis parle de nos 4 stratégies de retrieval et comment à la fin fusionne tou en introduisant la figure qui présente les stratégies de retrieval. 
Pour ce dernier paragraphe parle rapidement des technologies et models utilisés : pour els 4 nodes, le chunking y a rien de spécial just chunking, pour les themes dis que c'est extraits aver bertopic puis nommé grace à un LLM, pour les entity c'est extrait avec spacy. Pour les aretes ne détaille rien. Pour les stratégies, dans la 1 dis juste que c'est semantic retrieval et que avec 4 c'est des stratégies hors graph c'est topic dist avec JS divergence  que 2 et 3 c'est un parcours de graph 

\paragraph{Technical components.}
\begin{itemize}[itemsep=0pt,topsep=3pt]
    \item \textbf{Chunking}: Standard sentence-based chunking with character-level soft limits
    \item \textbf{Theme extraction}: BERTopic for topic modeling, LLM (Ollama) for descriptive labels
    \item \textbf{Entity extraction}: spaCy NER with filtering and normalization
    \item \textbf{Edge weighting}: TF-IDF with sigmoid for MENTIONS, PMI with sigmoid for CO\_OCCURS, cosine similarity for SEMANTIC\_SIM, exponential decay for TEMPORAL\_DIST
\end{itemize}

ces détails techniques ajoute les directements dans les paragraphe pour les noeuds et les stratégies de retrieval 
pas besoin de faire un paragraphe à part juste mentionne le quand tu traites ces cas là 

\paragraph{Technical components.}
\begin{itemize}[itemsep=0pt,topsep=3pt]
    \item \textbf{Chunking}: Standard sentence-based chunking with character-level soft limits
    \item \textbf{Theme extraction}: BERTopic for topic modeling, LLM (Ollama) for descriptive labels
    \item \textbf{Entity extraction}: spaCy NER with filtering and normalization
    \item \textbf{Edge weighting}: TF-IDF with sigmoid for MENTIONS, PMI with sigmoid for CO\_OCCURS, cosine similarity for SEMANTIC\_SIM, exponential decay for TEMPORAL\_DIST
\end{itemize}

ces détails techniques ajoute les directements dans les paragraphe pour les noeuds et les stratégies de retrieval 
pas besoin de faire un paragraphe à part juste mentionne le quand tu traites ces cas là 

Ensuite pour l'introduction fais juste une partie introduction pas besoin de faire des sous parties, garde les bullet list uniquement pour les noeuds, les aretes et les stratégies de retrieval enlève pour le reste et prioritise juste des paragraphes et des retours à la ligne voire des sauts de lignes si nécesaire. Détaille un tout petit plus ce que font les arêtes juste déjà de qui vers qui et ce qu'elles représentent en terme de significiation 

Ensuite touche pas à la section related work 

passons ensuite à la partie 3 qui sera la construction des noeuds fais une sous partie par noeud
explique en détail ici en t'aidant du code explique précisément et formule sans aucune ambiguité en donnant le plus de détail possible chaque transformation et comment on a chaque type de noeud (ne mets pas de snippet du code)
avec les mêmes directives fais la partie 4 qui sera construction des aretes avec pareil une sous partie par type d'arête (regroupe semantic sim pour entité et theme dans la même sous partie)
et ensuite partie 5 qui sera sur les stratégies de retrieval détaille bien ave cune sous partie par stratégie et une sous partie à lafin pour la fusion
ensuite partie 6 avec l'exemple préliminaire 
Fini avec une partie 7 qui sera la conclusion et ce qu'on retient 