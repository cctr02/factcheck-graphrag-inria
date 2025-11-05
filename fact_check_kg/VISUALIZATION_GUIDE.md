# Guide de Visualisation du Knowledge Graph

## 🎨 Comment créer et explorer les visualisations

### Étape 1 : Créer les visualisations

Après avoir construit le graphe avec `build_graph.py`, créez les visualisations :

```bash
cd fact_check_kg

# 1. Vue complète hiérarchique (175-200 nœuds)
uv run python visualize_hierarchical.py --mode full

# 2. Vue centrée sur une entité riche (Nigeria, Myanmar, etc.)
uv run python visualize_hierarchical.py --mode entity --entity nigeria

# 3. Statistiques textuelles (rapide)
uv run python visualize_graph.py --mode stats
```

---

## 📂 Fichiers HTML générés

Après exécution, vous aurez :

```
fact_check_kg/
├── graph_full_hierarchical.html           # Vue d'ensemble
├── graph_hierarchical_nigeria.html        # Vue centrée Nigeria
├── graph_hierarchical_[entity].html       # Autres vues centrées
```

---

## 🎯 Quel HTML regarder ?

### **Pour comprendre la structure globale**
→ **`graph_full_hierarchical.html`** ⭐⭐⭐

**Ce que vous verrez :**
- 10 documents en haut (bleu □)
- Leurs chunks au milieu (jaune ■)
- Thèmes et entités en bas (violet ◆ et vert ●)
- Layout hiérarchique vertical

**Quand l'utiliser :**
- Première exploration du graphe
- Comprendre la distribution des thèmes
- Voir les documents les plus connectés

---

### **Pour explorer en détail une entité**
→ **`graph_hierarchical_nigeria.html`** ⭐⭐⭐

**Ce que vous verrez :**
- ★ Nigeria au centre (rouge, étoile)
- ◆ 3 thèmes liés (violet, diamants) avec labels LLM
- ■ 15 chunks mentionnant Nigeria (jaune, rectangles)
- ● 10 entités co-occurentes (vert, cercles)
- □ 5 documents sources (bleu, carrés)

**Quand l'utiliser :**
- Explorer un concept précis (Nigeria, Facebook, vaccine, etc.)
- Comprendre les connexions d'une entité
- Voir les thèmes associés à un sujet

**Créer pour d'autres entités :**
```bash
# Myanmar
uv run python visualize_hierarchical.py --mode entity --entity myanmar

# Vaccine
uv run python visualize_hierarchical.py --mode entity --entity vaccine

# Biden
uv run python visualize_hierarchical.py --mode entity --entity biden
```

---

## 🔍 Comment lire les visualisations

### **Formes des nœuds :**
- ★ **Étoile** : Entité centrale (focus de la vue)
- ◆ **Diamant** : Thème (topic modeling)
- ■ **Rectangle** : Chunk de texte
- ● **Cercle** : Entité liée
- □ **Carré** : Document source

### **Couleurs des nœuds :**
- 🔴 **Rouge** : Entité centrale
- 🟣 **Violet** : Thèmes
- 🟡 **Jaune** : Chunks
- 🟢 **Vert** : Entités liées
- 🔵 **Bleu** : Documents

### **Types d'arêtes :**
- Vert (MENTIONS) : Chunk mentionne une entité
- Violet (HAS_TOPIC) : Chunk appartient à un thème
- Rouge (CO_OCCURS) : Deux entités apparaissent ensemble
- Bleu (SEMANTIC_SIM) : Similarité sémantique entre entités
- Orange (RELATED_TO) : Entité liée à un thème
- Gris (CONTAINS) : Document contient un chunk

---

## 💡 Conseils d'utilisation

### **Dans le navigateur :**

1. **Hover** sur un nœud → Voir détails (type, text preview, degree)
2. **Clic sur nœud** → Sélectionner et voir connexions
3. **Zoom** : Molette de souris
4. **Pan** : Clic + glisser
5. **Physique** : Bouton "Physics" en bas à gauche pour ajuster

### **Problèmes courants :**

**Trop de nœuds (illisible) ?**
```bash
# Limiter le nombre de nœuds affichés
uv run python visualize_hierarchical.py --mode full --max-nodes 100
```

**Thèmes pas clairs ?**
```bash
# Inspecter les thèmes en texte
uv run python inspect_topics.py
```

**Entité introuvable ?**
```bash
# Vérifier les entités disponibles
uv run python visualize_graph.py --mode stats
# Voir "Top 15 entities by degree"
```

---

## 📊 Workflow recommandé

### **1. Vue d'ensemble (5 min)**
```bash
# Statistiques
uv run python visualize_graph.py --mode stats

# Vue hiérarchique complète
uv run python visualize_hierarchical.py --mode full
```
→ Ouvrir `graph_full_hierarchical.html`

### **2. Explorer un thème spécifique (10 min)**

Choisir une entité intéressante des stats, par exemple "nigeria" :
```bash
uv run python visualize_hierarchical.py --mode entity --entity nigeria
```
→ Ouvrir `graph_hierarchical_nigeria.html`

### **3. Inspecter les thèmes (5 min)**
```bash
uv run python inspect_topics.py
```
→ Voir les 63 thèmes avec keywords et labels LLM

### **4. Tester le retrieval (10 min)**
```bash
# Avec un claim lié à l'entité explorée
uv run python test_retrieval.py --claim "Nigeria fact check images"
```

---

## 🎯 Résumé : Quel fichier pour quel usage ?

| Besoin | Fichier HTML | Commande |
|--------|-------------|----------|
| **Vue d'ensemble** | `graph_full_hierarchical.html` | `visualize_hierarchical.py --mode full` |
| **Explorer Nigeria** | `graph_hierarchical_nigeria.html` | `visualize_hierarchical.py --entity nigeria` |
| **Explorer Myanmar** | `graph_hierarchical_myanmar.html` | `visualize_hierarchical.py --entity myanmar` |
| **Explorer vaccine** | `graph_hierarchical_vaccine.html` | `visualize_hierarchical.py --entity vaccine` |
| **Statistiques** | - (terminal) | `visualize_graph.py --mode stats` |
| **Thèmes** | - (terminal) | `inspect_topics.py` |

---

## 🚀 Exemples de visualisations intéressantes

### **Entités avec beaucoup de connexions (riches) :**
```bash
# AFP (organisation de fact-checking)
uv run python visualize_hierarchical.py --entity afp

# Facebook (plateforme de diffusion)
uv run python visualize_hierarchical.py --entity facebook

# COVID/Coronavirus
uv run python visualize_hierarchical.py --entity coronavirus
```

### **Entités géographiques :**
```bash
uv run python visualize_hierarchical.py --entity nigeria
uv run python visualize_hierarchical.py --entity myanmar
uv run python visualize_hierarchical.py --entity kenya
```

### **Thématiques :**
```bash
uv run python visualize_hierarchical.py --entity vaccine
uv run python visualize_hierarchical.py --entity election
uv run python visualize_hierarchical.py --entity tax
```

---

## 📝 Notes importantes

- Les visualisations montrent des **sous-graphes** du graphe complet (sinon illisible)
- La vue `full` affiche ~200 nœuds parmi les 6000+ du graphe total
- Les vues `entity` montrent 2-3 niveaux de voisinage autour d'une entité
- Les labels des thèmes sont générés par **Llama 3.1** via Ollama

---

## ✅ Checklist rapide

Après avoir construit le graphe avec 500 documents :

- [ ] Générer vue complète : `visualize_hierarchical.py --mode full`
- [ ] Voir stats : `visualize_graph.py --mode stats`
- [ ] Choisir 2-3 entités intéressantes des stats
- [ ] Générer vues centrées sur ces entités
- [ ] Inspecter thèmes : `inspect_topics.py`
- [ ] Ouvrir les HTML dans le navigateur
- [ ] Tester retrieval avec claims liés

**HTML principal à regarder : `graph_hierarchical_nigeria.html`** (ou autre entité riche)

---

Bon graphe exploration ! 🎨
