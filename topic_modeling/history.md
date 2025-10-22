## 1st Run
python topic_modeling/topic_modeling2.py --csv data/english_fact_checks.csv --topics 12 --out topics_run --log-level INFO --progress

---

* `max_features=30000` → bon compromis pour TF-IDF (au-delà, mémoire monte vite).
* `n_topics=12` → équilibré, mais certains doublons entre thèmes.
* `LDA perplexity = 3492` (→ correct pour des textes journalistiques, mais un peu élevé)
* `NMF MSE = 3.1e-05` → excellente reconstruction (modèle stable).
* Temps total ~11 min → normal vu la taille du corpus.

### 🧩 2. Qualité des topics

Sur le fichier `topics_run_nmf_top_terms.csv` :

* **Très bons clusters** : *Covid/santé*, *élections US*, *fact-checking réseaux sociaux*, *loi/immigration*, *économie/taxes*.
* **Chevauchements notables** :

  * Politique US : deux topics (Trump/Biden vs Trump/Obama/Clinton) → redondant.
  * Médias / Réseaux sociaux : deux topics proches (posts/photos vs vidéos/screenshots).
  * Santé : un doublon entre *Covid* et *Health/UK data*.
* **Topics génériques** : un ou deux regroupent des termes vagues (« world », « show », « years », « new ») → souvent bruit.

### ⚙️ 3. Paramètres actuels

| Paramètre            | Effet                       | Recommandation                                                          |
| -------------------- | --------------------------- | ----------------------------------------------------------------------- |
| `n_topics=12`        | Trop de chevauchement       | Tester **10** ou **14**                                                 |
| `max_df=0.95`        | Bon, mais garde du bruit    | Baisser à **0.90** pour filtrer termes trop fréquents                   |
| `min_df=2`           | Bas pour corpus de 38k docs | Monter à **5** pour éliminer les mots rares/bruit                       |
| `max_features=30000` | Suffisant                   | Conserver                                                               |
| `ngram_range=(1,1)`  | Manque de collocations      | Passer à **(1,2)** pour capturer « social media », « fact check », etc. |

---

## 🧹 Nettoyage linguistique

### Stopwords

Les termes [`news`, `article`, `story`, `would`, `people`, `amp`, `new` ]reviennent souvent sans valeur thématique.
➡️ Ajoute-les dans `build_stopwords_english()` :

```python
extra |= {"news", "article", "story", "would", "amp", "people", "new"}
```

### Autres nettoyages possibles

* Supprimer `amp` (artefact HTML)
* Fusionner *title* et *body* avec un séparateur plus marqué (`". "` est bien)
* Si tu veux renforcer la cohérence des thèmes, applique un **lemmatizer** (`nltk.WordNetLemmatizer`).

---

## 🔧 Recommandations pratiques

1. **Paramètres à modifier :**

   ```bash
   --topics 10
   --max-df 0.90
   --min-df 5
   --ngrams 1,2
   ```
2. **Stopwords à enrichir** (comme ci-dessus).
3. **Comparer LDA vs NMF** : garde les deux exports, mais NMF sera plus net pour ton corpus journalistique.
4. **Ajouter un indicateur de diversité** (topic diversity = % de mots uniques dans top-N) → pour mesurer objectivement si K=10 ou 14 est meilleur.
5. **Visualisation** : une heatmap `doc_topic` sur 100 échantillons t’aidera à repérer la dominance thématique.

---

## 🧭 En résumé

| Aspect           | État actuel           | Recommandation              |
| ---------------- | --------------------- | --------------------------- |
| Modèle           | Fonctionne très bien  | ✅ conserver                 |
| Nombre de topics | 12                    | tester 10 et 14             |
| Stopwords        | incomplets            | enrichir liste              |
| n-grams          | seulement unigrams    | activer bigrams             |
| min_df / max_df  | tolérants             | durcir (5 / 0.90)           |
| Interprétation   | bonne mais redondante | regrouper thèmes similaires |

---

Souhaites-tu que je te fasse directement le **script de recalibrage** (avec stopwords étendus, bigrams et topic diversity automatique) pour tester K=10/14 côte à côte ?


## 2nd run
python topic_modeling/topic_modeling2.py --csv data/english_fact_checks.csv   --topics 10  --max-df 0.90  --min-df 5  --max-features 30000  --ngrams 1,2  --out topics_run_v2  --log-level INFO  --progress
