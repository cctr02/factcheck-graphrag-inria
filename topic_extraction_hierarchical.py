#!/usr/bin/env python3
"""
Extraction de thèmes HIÉRARCHIQUES avec BERTopic - Surthèmes et Sous-thèmes.

Ce fichier utilise la hiérarchie naturelle de HDBSCAN pour créer:
- Topics (niveau fin): ex. "COVID in Australia", "COVID in America"
- Supertopics (niveau supérieur): ex. "COVID Pandemic"

DIFFÉRENCES avec topic_extraction_isolated.py:
-------------------------------------------
1. Utilise topic_model.hierarchical_topics() pour remonter dans l'arbre HDBSCAN
2. Crée une mapping topic → supertopic
3. Double assignation pour chaque chunk: topics ET supertopics
4. HTML enrichi montrant les deux niveaux hiérarchiques

MÉTHODE:
--------
- HDBSCAN crée un dendrogram complet (arbre de clusters)
- BERTopic.hierarchical_topics() fusionne progressivement les topics similaires
- On sélectionne un niveau intermédiaire pour obtenir les surthèmes
- Distance de fusion: basée sur cosine similarity entre représentations c-TF-IDF
"""

import sys
import time
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from datetime import datetime
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet', quiet=True)

# Ajouter le répertoire du script au path pour les imports locaux (même que build_graph)
sys.path.insert(0, str(Path(__file__).parent))
from fact_check_kg.data_loader import load_and_chunk_documents


def chunk_text_for_topics(text: str, min_chunk_size: int = 400, max_chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into chunks with a MINIMUM size for topic modeling.
    (Identique à topic_extraction_isolated.py)
    """
    import re

    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_length = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_length = len(sentence)

        current_chunk.append(sentence)
        current_length += sentence_length + (1 if current_length > 0 else 0)

        should_create_chunk = False

        if current_length >= max_chunk_size:
            should_create_chunk = True
        elif current_length >= min_chunk_size and i == len(sentences) - 1:
            should_create_chunk = True
        elif current_length >= min_chunk_size and i < len(sentences) - 1:
            next_sentence_length = len(sentences[i + 1])
            if current_length + next_sentence_length + 1 > max_chunk_size:
                should_create_chunk = True

        if should_create_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunk_text:
                chunks.append(chunk_text)

            if i < len(sentences) - 1:
                overlap_sentences = []
                overlap_length = 0

                for s in reversed(current_chunk):
                    if overlap_length + len(s) + 1 <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + 1
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length
            else:
                current_chunk = []
                current_length = 0

        i += 1

    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) < min_chunk_size and len(chunks) > 0:
            chunks[-1] = chunks[-1] + ' ' + chunk_text
        elif chunk_text:
            chunks.append(chunk_text)

    return chunks


class HierarchicalTopicExtractor:
    """
    Classe pour extraire des thèmes HIÉRARCHIQUES avec BERTopic.
    Crée à la fois des topics (niveau fin) et des supertopics (niveau supérieur).
    """

    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        topic_keywords_initial: int = 40,
        topic_keywords_final: int = 15,
        logger: Optional[logging.Logger] = None,
        llm_endpoint: str = "http://localhost:11434/api/generate",
        llm_model: str = "llama3.1:latest",
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 24,
        llm_timeout: int = 120
    ):
        self.embedding_model = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')
        self.topic_keywords_initial = topic_keywords_initial
        self.train_docs = None
        self.topic_keywords_final = topic_keywords_final
        self.logger = logger or logging.getLogger(__name__)
        self.lemmatizer = WordNetLemmatizer()
        self.topic_model = None
        self.hierarchical_topics_df = None
        self.topic_to_supertopic = {}

        # Ollama / Llama
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_timeout = llm_timeout


    def build_topic_model(
        self,
        chunk_texts: List[str],
        min_topic_size: Optional[int] = None,
        nr_topics: Optional[int] = None
    ) -> Tuple[BERTopic, List[int], np.ndarray]:
        """
        Construit un modèle de topics avec hiérarchie.
        """
        start_time = time.time()
        self.logger.info("Training BERTopic model with hierarchical structure...")
        self.logger.info(f"Computing embeddings for {len(chunk_texts)} chunks...")

        if min_topic_size is None:
            min_topic_size = 10
        self.logger.info(f"Using min_cluster_size={min_topic_size} for {len(chunk_texts)} chunks")

        # CountVectorizer
        min_df_value = min(2, max(1, len(chunk_texts) // 5))
        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=min_df_value,
            ngram_range=(1, 3),
            max_features=1000
        )

        # UMAP
        n_samples = len(chunk_texts)
        max_components = max(2, n_samples - 2)
        n_components = min(5, max_components)
        max_neighbors = max(2, n_samples - 2)
        n_neighbors_umap = min(15, max_neighbors)

        self.logger.info(f"UMAP config: n_samples={n_samples}, n_components={n_components}, n_neighbors={n_neighbors_umap}")

        umap_model = UMAP(
            n_neighbors=n_neighbors_umap,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            verbose=True
        )

        # HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            core_dist_n_jobs=1
        )

        # BERTopic
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=nr_topics,
            calculate_probabilities=True,
            verbose=True
        )

        self.logger.info(f"Starting BERTopic pipeline on {len(chunk_texts)} text chunks...")

        # Fit and transform
        topics, probabilities = topic_model.fit_transform(chunk_texts)
        self.train_docs = chunk_texts


        self.topic_model = topic_model

        elapsed = time.time() - start_time
        self.logger.info(f"✓ Topic model trained in {elapsed:.2f}s")

        return topic_model, topics, probabilities

    def build_hierarchical_structure(self, nr_supertopics: Optional[int] = None) -> Dict:
        """
        Construit la structure hiérarchique: topics → supertopics.

        Utilise BERTopic.hierarchical_topics() pour remonter dans l'arbre HDBSCAN.

        Args:
            nr_supertopics: Nombre cible de supertopics (None = auto, environ 30-50% des topics)

        Returns:
            Dict contenant la hiérarchie complète
        """
        if self.topic_model is None:
            raise ValueError("Topic model not trained yet. Call build_topic_model() first.")

        self.logger.info("\n" + "="*80)
        self.logger.info("BUILDING HIERARCHICAL STRUCTURE (Topics → Supertopics)")
        self.logger.info("="*80)

        # Obtenir tous les topics (sans outliers)
        all_topics = [t for t in self.topic_model.get_topics().keys() if t != -1]
        n_topics = len(all_topics)

        self.logger.info(f"Starting with {n_topics} fine-grained topics")

        # Calculer le nombre cible de supertopics
        if nr_supertopics is None:
            # Par défaut: viser environ 40% du nombre de topics
            nr_supertopics = max(3, int(n_topics * 0.4))

        self.logger.info(f"Target: {nr_supertopics} supertopics (reducing by ~{100*(1-nr_supertopics/n_topics):.0f}%)")

        # Obtenir la hiérarchie complète via BERTopic
        # hierarchical_topics() utilise la distance cosine entre représentations c-TF-IDF
        self.logger.info("Computing hierarchical merging via c-TF-IDF similarity...")
        hierarchical_topics = self.topic_model.hierarchical_topics(self.train_docs)
        self.logger.info(f"hierarchical_topics columns: {list(hierarchical_topics.columns)}")
        self.logger.info(f"hierarchical_topics head:\n{hierarchical_topics.head(3)}")


        self.hierarchical_topics_df = hierarchical_topics
        self.logger.info(f"✓ Hierarchical structure computed: {len(hierarchical_topics)} merge steps")

        # Trouver le bon niveau de coupe pour obtenir nr_supertopics
        # La hierarchical_topics table montre les fusions progressives
        # Format: Parent_ID, Topics_left, Topics_right, Distance

        # Stratégie: partir du nombre initial de topics et fusionner jusqu'à atteindre nr_supertopics
        n_merges_needed = n_topics - nr_supertopics

        self.logger.info(f"Will perform {n_merges_needed} merges to go from {n_topics} → {nr_supertopics} supertopics")

        # Créer le mapping topic → supertopic
        # Au départ, chaque topic est son propre supertopic
        topic_to_supertopic = {tid: tid for tid in all_topics}

        # Nombre de merges nécessaires
        n_merges_needed = n_topics - nr_supertopics
        n_merges_needed = max(0, n_merges_needed)

        # Mapping topic -> supertopic (par défaut: lui-même)
        topic_to_supertopic = {tid: tid for tid in all_topics}

        # Dans cette version BERTopic:
        # - Parent_ID = id du cluster fusionné
        # - Topics = liste des topics (feuilles) contenus dans ce cluster
        # On prend les n_merges_needed premières fusions (plus proches / plus petites distances en général)
        for idx in range(min(n_merges_needed, len(hierarchical_topics))):
            row = hierarchical_topics.iloc[idx]

            topics_in_parent = row["Topics"]  # <-- liste de topics feuilles
            if isinstance(topics_in_parent, (int, np.integer)):
                topics_in_parent = [int(topics_in_parent)]
            else:
                topics_in_parent = [int(t) for t in topics_in_parent]

            # Choix d’un "supertopic id" stable: min des topics du groupe
            super_id = min(topics_in_parent)

            for tid in topics_in_parent:
                if tid in topic_to_supertopic:
                    topic_to_supertopic[tid] = super_id

        self.topic_to_supertopic = topic_to_supertopic


        # Statistiques
        unique_supertopics = set(topic_to_supertopic.values())
        self.logger.info(f"✓ Hierarchy created: {n_topics} topics → {len(unique_supertopics)} supertopics")

        # Afficher quelques exemples de regroupements
        supertopic_groups = {}
        for topic_id, supertopic_id in topic_to_supertopic.items():
            if supertopic_id not in supertopic_groups:
                supertopic_groups[supertopic_id] = []
            supertopic_groups[supertopic_id].append(topic_id)

        self.logger.info("\nExample supertopic groupings:")
        for super_id in sorted(supertopic_groups.keys())[:5]:
            topics = supertopic_groups[super_id]
            self.logger.info(f"  Supertopic {super_id}: contains {len(topics)} topics {topics}")

        return {
            'topic_to_supertopic': topic_to_supertopic,
            'supertopic_groups': supertopic_groups,
            'hierarchical_df': hierarchical_topics
        }

    def _process_topic_keywords(self, topic_words: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Pipeline:
        1. Prendre top-N keywords BERTopic
        2. Supprimer les keywords qui sont sous-chaînes d'autres (garder le plus long)
        3. Lemmatiser
        4. Fusionner les scores des formes équivalentes
        5. Garder top 15
        """
        if not topic_words:
            return []

        # Step 1 — garder top N init
        candidates = topic_words[:self.topic_keywords_initial]

        # Step 2 — substring dedup
        # Trier par longueur décroissante pour garder les plus spécifiques
        sorted_words = sorted(candidates, key=lambda x: len(x[0]), reverse=True)

        filtered = []
        for word, score in sorted_words:
            word_lower = word.lower()

            # skip si sous-chaîne d'un mot déjà gardé
            if any(word_lower in kept_word.lower() for kept_word, _ in filtered):
                continue

            filtered.append((word, score))

        # Step 3 — lemmatization
        lemma_scores: Dict[str, float] = {}

        for word, score in filtered:
            tokens = word.lower().split()
            lemmas = [self.lemmatizer.lemmatize(t) for t in tokens]
            lemma_phrase = " ".join(lemmas)

            lemma_scores[lemma_phrase] = lemma_scores.get(lemma_phrase, 0.0) + score

        # Step 4 — sort by score
        final_sorted = sorted(lemma_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 5 — top 15
        top_keywords = final_sorted[:self.topic_keywords_final]

        return top_keywords


    def get_topic_label(
        self,
        topic_words: List[Tuple[str, float]],
        use_llm: bool = False
    ) -> str:
        """Génère un label pour un topic (identique à isolated)."""
        processed_keywords = self._process_topic_keywords(topic_words)

        if use_llm:
            return self._label_topic_with_llm(processed_keywords)
        else:
            top_words = [word for word, score in processed_keywords]
            label = " ".join(top_words[:3])
            return label

    def _label_topic_with_llm(self, keywords: List[Tuple[str, float]]) -> str:
        """
        Utilise Llama via Ollama pour générer un label lisible (2-5 mots).
        """
        try:
            import requests

            keyword_list = ", ".join([f"{w} ({s:.3f})" for w, s in keywords[:10]])

            prompt = f"""You create short, readable topic labels.

    Keywords: {keyword_list}

    Rules:
    - 2 to 5 words maximum
    - Proper English (avoid stem fragments like "effici", "accuraci")
    - No quotes, no punctuation at the end
    - Output ONLY the label

    Label:"""

            r = requests.post(
                self.llm_endpoint,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.llm_temperature,
                        "num_predict": self.llm_max_tokens
                    }
                },
                timeout=self.llm_timeout
            )

            if r.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")

            label = (r.json().get("response") or "").strip()
            label = label.replace("\n", " ").strip().strip('"').strip("'")

            # garde-fous
            if not label:
                raise RuntimeError("Empty label returned by Ollama")

            words = label.split()
            if len(words) > 5:
                label = " ".join(words[:5])

            return label

        except Exception as e:
            self.logger.warning(f"LLM labeling failed (Ollama/Llama): {e}, fallback keywords")
            return " ".join([w for w, _ in keywords[:3]])
        
    def _label_supertopic_with_llm(
        self,
        child_topic_labels: List[str],
        aggregated_keywords: List[Tuple[str, float]],
    ) -> str:
        """
        Label de supertopic basé sur les labels des sous-topics + keywords agrégés.
        """
        try:
            import requests

            # On limite pour rester court (sinon prompt énorme)
            child_topic_labels = [lbl for lbl in child_topic_labels if lbl and lbl != "unknown"][:12]
            children_text = "\n".join([f"- {lbl}" for lbl in child_topic_labels])

            kw_text = ", ".join([kw for kw, _ in aggregated_keywords[:8]])

            prompt = f"""You create a broad theme label (supertopic) from its child topics.

    Child topic labels:
    {children_text}

    Helpful keywords: {kw_text}

    Rules:
    - 2 to 6 words maximum
    - Broad, not too specific
    - Proper English
    - Output ONLY the label (no quotes)

    Supertopic label:"""

            r = requests.post(
                self.llm_endpoint,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.llm_temperature,
                        "num_predict": max(self.llm_max_tokens, 32)
                    }
                },
                timeout=self.llm_timeout
            )

            if r.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")

            label = (r.json().get("response") or "").strip()
            label = label.replace("\n", " ").strip().strip('"').strip("'")

            if not label:
                raise RuntimeError("Empty supertopic label returned by Ollama")

            words = label.split()
            if len(words) > 6:
                label = " ".join(words[:6])

            return label

        except Exception as e:
            self.logger.warning(f"Supertopic LLM labeling failed: {e}, fallback keywords")
            return " ".join([kw for kw, _ in aggregated_keywords[:3]])



    def get_all_topics_info(self, use_llm_labels: bool = True) -> dict:
        """Récupère les infos de tous les topics (niveau fin)."""
        if self.topic_model is None:
            raise ValueError("Topic model not trained yet.")

        topics_info = {}
        topic_freq = self.topic_model.get_topic_freq()

        self.logger.info(f"Generating labels for {len(topic_freq)-1} topics (use_llm={use_llm_labels})...")

        for topic_id in topic_freq['Topic']:
            if topic_id == -1:
                continue

            topic_words = self.topic_model.get_topic(topic_id)
            if not topic_words:
                continue

            processed_keywords = self._process_topic_keywords(topic_words)
            label = self.get_topic_label(topic_words, use_llm=use_llm_labels)
            count = topic_freq[topic_freq['Topic'] == topic_id]['Count'].values[0]

            topics_info[topic_id] = {
                'label': label,
                'keywords': [w for w, s in processed_keywords],
                'keyword_scores': processed_keywords,
                'raw_keywords': topic_words[:10],
                'count': count,
                'supertopic': self.topic_to_supertopic.get(topic_id, topic_id)  # Lien vers supertopic
            }

        return topics_info

    def get_supertopics_info(self, topics_info: dict, use_llm_labels: bool = True) -> dict:
        """
        Crée les infos des supertopics en agrégeant les topics enfants.
        Génère un label de supertopic via LLM à partir des labels enfants + keywords agrégés.
        """
        self.logger.info("Creating supertopic labels and aggregating child topics...")

        # Regrouper les topics par supertopic
        supertopic_groups: Dict[int, List[int]] = {}
        for topic_id, info in topics_info.items():
            super_id = int(info.get("supertopic", topic_id))
            supertopic_groups.setdefault(super_id, []).append(int(topic_id))

        supertopics_info: Dict[int, dict] = {}

        for super_id, child_topics in supertopic_groups.items():
            # Agréger les keywords de tous les topics enfants
            all_keywords: Dict[str, float] = {}
            total_count = 0

            for topic_id in child_topics:
                topic_info = topics_info[topic_id]
                total_count += int(topic_info["count"])

                # Fusionner les keywords pondérés par score
                for kw, score in topic_info["keyword_scores"]:
                    all_keywords[kw] = all_keywords.get(kw, 0.0) + float(score)

            # Trier par score agrégé
            sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
            top_keywords = sorted_keywords[: self.topic_keywords_final]

            # Label du supertopic
            if use_llm_labels:
                child_labels = [topics_info[t]["label"] for t in child_topics if t in topics_info]
                label = self._label_supertopic_with_llm(child_labels, top_keywords)
            else:
                label = " ".join([kw for kw, _ in top_keywords[:3]])

            # Sauvegarder
            supertopics_info[int(super_id)] = {
                "label": label,
                "keywords": [kw for kw, _ in top_keywords],
                "keyword_scores": top_keywords,
                "child_topics": sorted(child_topics),
                "count": total_count,
                "n_topics": len(child_topics),
            }

        self.logger.info(f"✓ Created {len(supertopics_info)} supertopics")
        return supertopics_info


    def assign_topics_to_chunks(
        self,
        topics: List[int],
        probabilities: np.ndarray,
        top_k: int = 3,
        min_prob: float = 0.01
    ) -> List[List[Tuple[int, float]]]:
        """Assigne les top-K topics à chaque chunk (identique à isolated)."""
        assignments = []

        for idx in range(len(topics)):
            if probabilities is not None and len(probabilities.shape) == 2:
                chunk_probs = probabilities[idx]
                valid_indices = np.where(chunk_probs >= min_prob)[0]
                valid_probs = chunk_probs[valid_indices]

                top_indices = np.argsort(valid_probs)[-top_k:][::-1]
                chunk_assignments = [
                    (int(valid_indices[i]), float(valid_probs[i]))
                    for i in top_indices
                    if valid_indices[i] != -1
                ]

                chunk_assignments.sort(key=lambda x: x[1], reverse=True)
                assignments.append(chunk_assignments)
            else:
                assignments.append([(int(topics[idx]), 1.0)])

        return assignments

    def generate_hierarchical_html_report(
        self,
        chunk_texts: List[str],
        topics: List[int],
        probabilities: np.ndarray,
        topics_info: dict,
        supertopics_info: dict,
        output_dir: str = "topic_extraction_results",
        sample_chunks_indices: Optional[List[int]] = None,
        chunk_ids: Optional[List[str]] = None
    ) -> str:
        """
        Génère un rapport HTML hiérarchique montrant topics ET supertopics.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        
        topic_assignments = self.assign_topics_to_chunks(topics, probabilities, top_k=3)
        import random

        max_display = 1000
        all_indices = list(range(len(chunk_texts)))
        random.seed(42)  # reproductible
        display_indices = random.sample(all_indices, min(max_display, len(all_indices)))
        display_indices.sort()


        # Calculer les assignations de supertopics
        supertopic_assignments = []
        for chunk_topics in topic_assignments:
            # Agréger les probabilités par supertopic
            super_probs = {}
            for topic_id, prob in chunk_topics:
                super_id = topics_info.get(topic_id, {}).get('supertopic', topic_id)
                if super_id not in super_probs:
                    super_probs[super_id] = 0
                super_probs[super_id] += prob

            # Top-3 supertopics
            sorted_supers = sorted(super_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            supertopic_assignments.append(sorted_supers)

        # HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hierarchical Topic Extraction Report - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-box {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}

        /* Supertopics styling */
        .supertopics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .supertopic-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            color: white;
        }}
        .supertopic-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .supertopic-id {{
            font-size: 1.4em;
            font-weight: bold;
        }}
        .supertopic-count {{
            background-color: rgba(255,255,255,0.3);
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        .supertopic-label {{
            font-size: 1.2em;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .child-topics {{
            background-color: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }}
        .child-topic-item {{
            background-color: rgba(255,255,255,0.2);
            padding: 8px 12px;
            margin: 5px;
            border-radius: 5px;
            display: inline-block;
            font-size: 0.9em;
        }}

        /* Topics styling */
        .topics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .topic-card {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-top: 4px solid #3498db;
        }}
        .topic-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .topic-id {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .topic-count {{
            background-color: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
        .topic-label {{
            font-size: 1.1em;
            color: #27ae60;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .parent-supertopic {{
            font-size: 0.9em;
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 10px;
        }}
        .keywords {{
            margin-top: 10px;
        }}
        .keyword {{
            display: inline-block;
            background-color: #ecf0f1;
            padding: 5px 10px;
            margin: 3px;
            border-radius: 3px;
            font-size: 0.9em;
        }}

        /* Chunks styling */
        .chunks-section {{
            margin-top: 30px;
        }}
        .chunk-card {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chunk-header {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 2px solid #ecf0f1;
            font-size: 1.1em;
        }}
        .chunk-text {{
            color: #34495e;
            line-height: 1.6;
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .hierarchy-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }}
        .hierarchy-panel {{
            padding: 15px;
            border-radius: 8px;
        }}
        .supertopics-panel {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-left: 4px solid #667eea;
        }}
        .topics-panel {{
            background-color: #e8f5e9;
            border-left: 4px solid #27ae60;
        }}
        .panel-title {{
            font-weight: bold;
            font-size: 1.05em;
            margin-bottom: 12px;
            color: #2c3e50;
        }}
        .assignment-item {{
            padding: 10px;
            margin: 8px 0;
            border-radius: 5px;
            background-color: rgba(255,255,255,0.7);
        }}
        .assignment-name {{
            font-weight: bold;
            font-size: 0.95em;
        }}
        .assignment-prob {{
            color: #7f8c8d;
            font-size: 0.85em;
            margin-top: 3px;
        }}
        .progress-bar {{
            height: 6px;
            background-color: #ecf0f1;
            border-radius: 3px;
            margin-top: 5px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        .progress-fill-super {{
            background: linear-gradient(90deg, #667eea, #764ba2);
        }}
        .progress-fill-topic {{
            background: linear-gradient(90deg, #27ae60, #2ecc71);
        }}
        .timestamp {{
            text-align: right;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <h1>🌳 Hierarchical Topic Extraction Report</h1>

    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="summary-stats">
            <div class="stat-box">
                <div class="stat-label">Total Chunks Analyzed</div>
                <div class="stat-value">{len(chunk_texts)}</div>
            </div>
            <div class="stat-box" style="border-left-color: #667eea;">
                <div class="stat-label">Supertopics (Level 2)</div>
                <div class="stat-value">{len(supertopics_info)}</div>
            </div>
            <div class="stat-box" style="border-left-color: #27ae60;">
                <div class="stat-label">Topics (Level 1)</div>
                <div class="stat-value">{len(topics_info)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Hierarchy Ratio</div>
                <div class="stat-value">{len(topics_info) / max(len(supertopics_info), 1):.1f}:1</div>
            </div>
        </div>
    </div>

    <h2>🌟 Supertopics (Level 2 - Broad Themes)</h2>
    <div class="supertopics-grid">
"""

        # Ajouter chaque supertopic
        for super_id, super_info in sorted(supertopics_info.items(), key=lambda x: x[1]['count'], reverse=True):
            html += f"""
        <div class="supertopic-card">
            <div class="supertopic-header">
                <div class="supertopic-id">Supertopic {super_id}</div>
                <div class="supertopic-count">{super_info['count']} docs</div>
            </div>
            <div class="supertopic-label">"{super_info['label']}"</div>
            <div style="font-size: 0.9em; margin-bottom: 10px;">
                Keywords: {', '.join(super_info['keywords'][:5])}
            </div>
            <div class="child-topics">
                <div style="font-weight: bold; margin-bottom: 8px; font-size: 0.9em;">
                    Contains {super_info['n_topics']} topics:
                </div>
"""
            for child_id in super_info['child_topics']:
                child_label = topics_info.get(child_id, {}).get('label', 'unknown')
                html += f'                <span class="child-topic-item">Topic {child_id}: {child_label}</span>\n'

            html += """            </div>
        </div>
"""

        html += f"""    </div>

    <h2>📊 Topics (Level 1 - Fine-Grained)</h2>
    <div class="topics-grid">
"""

        # Ajouter chaque topic
        for topic_id, info in sorted(topics_info.items(), key=lambda x: x[1]['count'], reverse=True):
            super_id = info['supertopic']
            super_label = supertopics_info.get(super_id, {}).get('label', 'unknown')

            html += f"""
        <div class="topic-card">
            <div class="topic-header">
                <div class="topic-id">Topic {topic_id}</div>
                <div class="topic-count">{info['count']} docs</div>
            </div>
            <div class="parent-supertopic">↑ Supertopic {super_id}: {super_label}</div>
            <div class="topic-label">"{info['label']}"</div>
            <div class="keywords">
"""
            for keyword in info['keywords']:
                html += f'                <span class="keyword">{keyword}</span>\n'

            html += """            </div>
        </div>
"""

        html += """    </div>

    <h2>📝 All Chunk Assignments (Topics + Supertopics)</h2>
    <div class="chunks-section">
"""

        # Assignations pour tous les chunks
        for idx in display_indices:
            chunk_topics = topic_assignments[idx]
            chunk_supertopics = supertopic_assignments[idx]
            chunk_id_display = chunk_ids[idx] if chunk_ids and idx < len(chunk_ids) else ""

            html += f"""
        <div class="chunk-card">
            <div class="chunk-header">Chunk {idx + 1} <span style="color: #7f8c8d; font-weight: normal; font-size: 0.9em;">(ID: {chunk_id_display})</span></div>
            <div class="chunk-text">{chunk_texts[idx]}...</div>

            <div class="hierarchy-section">
                <div class="hierarchy-panel supertopics-panel">
                    <div class="panel-title">🌟 Supertopics:</div>
"""
            for super_id, prob in chunk_supertopics:
                super_label = supertopics_info.get(super_id, {}).get('label', 'unknown')
                prob_percent = prob * 100
                html += f"""
                    <div class="assignment-item">
                        <div class="assignment-name">Supertopic {super_id}: {super_label}</div>
                        <div class="assignment-prob">Probability: {prob:.3f} ({prob_percent:.1f}%)</div>
                        <div class="progress-bar">
                            <div class="progress-fill progress-fill-super" style="width: {min(prob_percent, 100)}%"></div>
                        </div>
                    </div>
"""

            html += """                </div>

                <div class="hierarchy-panel topics-panel">
                    <div class="panel-title">📊 Topics:</div>
"""
            for topic_id, prob in chunk_topics:
                topic_label = topics_info.get(topic_id, {}).get('label', 'unknown')
                prob_percent = prob * 100
                html += f"""
                    <div class="assignment-item">
                        <div class="assignment-name">Topic {topic_id}: {topic_label}</div>
                        <div class="assignment-prob">Probability: {prob:.3f} ({prob_percent:.1f}%)</div>
                        <div class="progress-bar">
                            <div class="progress-fill progress-fill-topic" style="width: {min(prob_percent, 100)}%"></div>
                        </div>
                    </div>
"""

            html += """                </div>
            </div>
        </div>
"""

        
        html += f"""
    </div>

    <div class="timestamp">
        Generated on {timestamp}<br>
        Hierarchical Structure: {len(topics_info)} topics grouped into {len(supertopics_info)} supertopics<br>
    </div>
</body>
</html>
"""

        # Sauvegarder
        output_file = output_path / f"hierarchical_topics_report_{timestamp}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        return str(output_file)


def main():
    """
    Exemple d'utilisation de HierarchicalTopicExtractor.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Configuration du chargement (même pipeline que build_graph.py)
    MAX_DOCS = 1000  # 1000 premières lignes du CSV (≈ 9 200 chunks)
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # Chemin vers le CSV (même fichier que build_graph)
    csv_path = str(Path(__file__).parent / "data" / "english_fact_checks.csv")

    logger.info(f"Loading documents from CSV: {csv_path}")
    logger.info(f"Max documents: {MAX_DOCS} (None = tous)")

    # Charger et chunker (même fonction que build_graph → data_loader)
    chunks, n_docs = load_and_chunk_documents(
        csv_path=csv_path,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        max_docs=MAX_DOCS,
        logger=logger
    )

    # Extraire textes et IDs (format: "chunk::{doc_id}::{chunk_idx}")
    sample_texts = [c['text'] for c in chunks]
    chunk_ids = [c['chunk_id'] for c in chunks]

    logger.info(f"Loaded {len(sample_texts)} chunks from {n_docs} documents")
    logger.info(f"Example chunk IDs: {chunk_ids[:5]}")

    # Initialiser l'extracteur hiérarchique
    logger.info("Initializing HierarchicalTopicExtractor...")
    extractor = HierarchicalTopicExtractor(logger=logger)

    # Entraîner le modèle
    logger.info("\nTraining hierarchical topic model...")
    topic_model, topics, probabilities = extractor.build_topic_model(
        sample_texts,
        min_topic_size=None,  # Auto-adaptive
        nr_topics=None
    )

    # Construire la hiérarchie
    hierarchy = extractor.build_hierarchical_structure(nr_supertopics=None)  # Auto: ~40% reduction

    # Obtenir les infos des topics
    logger.info("\n" + "="*80)
    logger.info("EXTRACTING TOPIC AND SUPERTOPIC INFORMATION")
    logger.info("="*80)

    topics_info = extractor.get_all_topics_info(use_llm_labels=True)
    supertopics_info = extractor.get_supertopics_info(topics_info, use_llm_labels=True)

    # Afficher quelques exemples
    logger.info(f"\n✓ Created {len(topics_info)} topics and {len(supertopics_info)} supertopics")

    logger.info("\nExample supertopics:")
    for super_id in sorted(supertopics_info.keys())[:3]:
        super_info = supertopics_info[super_id]
        logger.info(f"\nSupertopic {super_id}: {super_info['label']}")
        logger.info(f"  Contains {super_info['n_topics']} topics: {super_info['child_topics']}")
        logger.info(f"  Keywords: {', '.join(super_info['keywords'])}")

    # Générer le rapport HTML hiérarchique
    logger.info("\n" + "="*80)
    logger.info("GENERATING HIERARCHICAL HTML REPORT")
    logger.info("="*80)

    fixed_sample_chunks = [10, 50, 90] if len(sample_texts) > 90 else [0, len(sample_texts)//2, len(sample_texts)-1]
    logger.info(f"\nFixed sample chunks for detailed analysis: {fixed_sample_chunks}")

    output_file = extractor.generate_hierarchical_html_report(
        chunk_texts=sample_texts,
        topics=topics,
        probabilities=probabilities,
        topics_info=topics_info,
        supertopics_info=supertopics_info,
        output_dir="topic_extraction_results",
        sample_chunks_indices=fixed_sample_chunks,
        chunk_ids=chunk_ids
    )

    logger.info(f"\n✓ Hierarchical HTML report saved to: {output_file}")
    logger.info(f"  Open this file in your browser to view the hierarchical topic structure.")


if __name__ == "__main__":
    main()
