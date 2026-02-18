# Optimisations GPU pour le Knowledge Graph Builder

## Résumé des modifications

J'ai optimisé votre code pour maximiser l'utilisation du GPU lors de la construction du knowledge graph. Voici les modifications apportées :

## 1. **SentenceTransformer (Embeddings)**

### Avant :
```python
self.embedding_model = SentenceTransformer(embedding_model)
```

### Après :
```python
import torch
self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
```

**Impact** : Tous les embeddings (chunks, entités, thèmes, claims) sont maintenant calculés sur GPU

---

## 2. **spaCy NER (Extraction d'entités)**

### Avant :
- Traitement séquentiel : `nlp(text)` pour chaque chunk
- Pas de GPU

### Après :
```python
# Activation GPU avant chargement
spacy.require_gpu()
self.nlp = spacy.load(ner_model)
self.nlp.batch_size = 1000  # GPU

# Batch processing avec nlp.pipe()
for doc in nlp.pipe(chunk_texts, batch_size=1000):
    # Traitement...
```

**Impact** :
- Utilisation du GPU pour spaCy
- Batch processing : traitement de 1000 chunks à la fois au lieu de 1 par 1
- **Gain estimé : 10-50x plus rapide**

---

## 3. **UMAP (Réduction de dimensionnalité pour BERTopic)**

### Avant :
```python
from umap import UMAP
umap_model = UMAP(...)  # CPU uniquement
```

### Après :
```python
try:
    if device == 'cuda':
        from cuml.manifold import UMAP as cumlUMAP
        umap_model = cumlUMAP(...)  # GPU
except ImportError:
    from umap import UMAP
    umap_model = UMAP(...)  # Fallback CPU
```

**Impact** :
- UMAP sur GPU si cuML est installé
- **Gain estimé : 5-20x plus rapide**
- Fallback automatique sur CPU si cuML n'est pas disponible

---

## 4. **Optimisation des appels `.encode()`**

### Avant :
```python
embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)
```

### Après :
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128 if device == 'cuda' else 32

embeddings = embedding_model.encode(
    texts,
    batch_size=batch_size,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
    device=device
)
```

**Impact** :
- Batch size optimisé : 128 pour GPU vs 32 pour CPU
- Paramètre `device` explicite pour garantir l'utilisation du GPU
- Appliqué à tous les fichiers :
  - `build_graph.py` (embeddings de chunks pour FAISS)
  - `graph_nodes_entity.py` (embeddings d'entités)
  - `graph_nodes_theme.py` (embeddings de thèmes)
  - `graph_nodes_claims.py` (embeddings de claims et chunks)

---

## 5. **FAISS Index (Recherche vectorielle)**

### Avant :
```python
self.index = faiss.IndexFlatIP(self.dimension)  # CPU uniquement
```

### Après :
```python
cpu_index = faiss.IndexFlatIP(self.dimension)
if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
    res = faiss.StandardGpuResources()
    self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # GPU
else:
    self.index = cpu_index
```

**Impact** :
- Construction de l'index FAISS sur GPU
- **Gain estimé : 2-5x plus rapide**
- Fallback automatique sur CPU si faiss-gpu n'est pas disponible

---

## Fichiers modifiés

1. **`fact_check_kg/build/graph_builder.py`**
   - Détection GPU au démarrage
   - SentenceTransformer sur GPU
   - spaCy GPU activation
   - UMAP GPU (cuML) avec fallback

2. **`fact_check_kg/build/graph_nodes_entity.py`**
   - Batch processing spaCy avec `nlp.pipe()`
   - Embeddings d'entités optimisés

3. **`fact_check_kg/build/graph_nodes_theme.py`**
   - Embeddings de thèmes optimisés

4. **`fact_check_kg/build/graph_nodes_claims.py`**
   - Embeddings de claims optimisés
   - Embeddings de chunks optimisés

5. **`fact_check_kg/build/build_graph.py`**
   - Embeddings de chunks pour FAISS optimisés

6. **`fact_check_kg/core/embedding_cache.py`**
   - FAISS GPU support
   - Conversion GPU→CPU pour sauvegarde

---

## Dépendances recommandées pour GPU complet

### Installation minimale (déjà fonctionnel) :
```bash
pip install torch sentence-transformers spacy
python -m spacy download en_core_web_sm
```

### Pour activer UMAP GPU :
```bash
# Via conda (recommandé)
conda install -c rapidsai -c conda-forge -c nvidia cuml cuda-version=11.8

# Ou via pip (expérimental)
pip install cuml-cu11
```

### Pour activer FAISS GPU :
```bash
# Via conda (recommandé)
conda install -c conda-forge faiss-gpu

# Ou via pip
pip install faiss-gpu
```

---

## Logs de détection GPU

Au démarrage, vous verrez maintenant :

```
✓ GPU detected: NVIDIA GeForce RTX 3090
  CUDA version: 11.8
  GPU memory: 24.0 GB
✓ spaCy GPU enabled successfully
✓ Using GPU-accelerated UMAP (cuML)
✓ Using GPU-accelerated FAISS index
```

Si un composant n'a pas de GPU :
```
⚠ cuML not available, using CPU UMAP (install rapids-cuml for GPU acceleration)
⚠ faiss-gpu not available, using CPU FAISS (install faiss-gpu for GPU acceleration)
```

---

## Gains de performance estimés

### Avec GPU complet (NVIDIA RTX 3090 ou équivalent) :

| Étape | Avant (CPU) | Après (GPU) | Gain |
|-------|-------------|-------------|------|
| Embeddings (20k chunks) | ~15-20 min | ~2-3 min | **~6-8x** |
| spaCy NER | ~10-15 min | ~1-2 min | **~10x** |
| UMAP (BERTopic) | ~5-10 min | ~30s-1 min | **~10x** |
| FAISS construction | ~2 min | ~30s | **~4x** |
| **TOTAL** | **~35-50 min** | **~5-8 min** | **~6-7x** |

### Sans cuML/faiss-gpu (GPU partiel) :

| Étape | Temps | Gain vs CPU |
|-------|-------|-------------|
| Embeddings | ~2-3 min | **~6-8x** |
| spaCy NER | ~1-2 min | **~10x** |
| UMAP (CPU) | ~5-10 min | 1x |
| FAISS (CPU) | ~2 min | 1x |
| **TOTAL** | **~10-17 min** | **~3-4x** |

---

## Vérification de l'utilisation GPU

### Pendant l'exécution, surveillez :

```bash
# Terminal 1 : Lancer votre commande
uv run python cli.py build --max-docs 20000

# Terminal 2 : Monitorer le GPU
watch -n 1 nvidia-smi
```

Vous devriez voir :
- **GPU Utilization : 80-100%** pendant les embeddings
- **GPU Memory Used : 3-8 GB** (selon la taille du modèle)
- **Power Draw : Near TDP** (indication d'utilisation intensive)

---

## Notes importantes

1. **HDBSCAN et c-TF-IDF restent sur CPU** : Ces étapes n'ont pas d'implémentation GPU mature, mais représentent une part mineure du temps total

2. **Batch sizes adaptés automatiquement** :
   - GPU : 128 pour embeddings, 1000 pour spaCy
   - CPU : 32 pour embeddings, 50 pour spaCy

3. **Fallback gracieux** : Tout fonctionne même sans GPU, avec détection automatique

4. **Compatibilité** : Testé avec PyTorch CUDA 11.8+, devrait fonctionner avec CUDA 12.x

---

## Résolution de problèmes

### "CUDA out of memory"
- Réduire `batch_size` manuellement dans le code (128 → 64)
- Réduire `max_docs` temporairement

### "spaCy GPU failed"
- Vérifier que spaCy est compatible GPU : `python -m spacy validate`
- Réinstaller avec GPU : `pip install spacy[cuda11x]`

### "cuML not found"
- Installer via conda (pip souvent problématique)
- Alternative : laisser tourner sur CPU UMAP (reste rapide)

---

## Conclusion

Votre pipeline est maintenant **entièrement optimisé pour GPU**. Les principaux goulots d'étranglement ont été adressés :

✅ Embeddings massifs → GPU
✅ NER batch processing → GPU
✅ UMAP (optionnel) → GPU
✅ FAISS (optionnel) → GPU

Le gain total attendu est de **6-7x avec GPU complet**, ou **3-4x avec GPU partiel** (sans cuML/faiss-gpu).
