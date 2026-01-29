#!/usr/bin/env python3
"""
Fichier isolé pour travailler sur l'extraction de thèmes avec BERTopic.

Ce fichier contient toute la logique d'extraction de topics actuellement utilisée
dans le projet, isolée pour permettre l'expérimentation.

FIXES APPLIED:
--------------
FIX #1: Adaptive min_cluster_size to prevent duplicate topics
    - Changed from fixed min_cluster_size=2 to adaptive: max(10, n_chunks // 100)
    - For 1000 docs: min_cluster_size=10 (1% of dataset)
    - Prevents over-fragmentation and exact duplicate topics

FIX #2: LLM-based topic labeling for readable labels
    - Added _label_topic_with_llm() function using OpenAI GPT-4o-mini
    - Replaces bizarre stemmed concatenations like "learn result accuraci effici ai"
    - With readable labels like "Machine Learning Results" or "AI Efficiency"
    - Set use_llm_labels=True in get_all_topics_info()

FIX #3: Hierarchical clustering explanation (no code change needed)
    - HDBSCAN uses cluster_selection_method='eom' (Excess of Mass)
    - Automatically selects hierarchy cut level via stability measure
    - Stability = persistence × number_of_points for each cluster
    - Selects clusters that maximize total stability across hierarchy
    - No manual cut level needed - fully automatic
"""

import time
import logging
from typing import List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from nltk.stem.snowball import SnowballStemmer


def chunk_text_for_topics(text: str, min_chunk_size: int = 400, max_chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into chunks with a MINIMUM size for topic modeling.

    Cette fonction garantit que chaque chunk a au moins min_chunk_size caractères
    pour fournir suffisamment de contexte à BERTopic.

    Args:
        text: Input text to chunk
        min_chunk_size: Minimum size for each chunk (default: 400)
        max_chunk_size: Maximum size for each chunk (default: 800)
        overlap: Overlap between chunks in characters (default: 100)

    Returns:
        List of text chunks (each chunk has at least min_chunk_size characters)
    """
    import re

    # Split text into sentences
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

        # Ajouter la phrase au chunk courant
        current_chunk.append(sentence)
        current_length += sentence_length + (1 if current_length > 0 else 0)

        # Vérifier si on peut créer un chunk
        # On crée un chunk SI :
        # 1. On a atteint min_chunk_size ET (on dépasse max_chunk_size OU c'est la fin)
        # 2. OU on a dépassé max_chunk_size de toute façon
        should_create_chunk = False

        if current_length >= max_chunk_size:
            # On a dépassé la taille max, créer le chunk
            should_create_chunk = True
        elif current_length >= min_chunk_size and i == len(sentences) - 1:
            # Dernier chunk et on a atteint le minimum
            should_create_chunk = True
        elif current_length >= min_chunk_size and i < len(sentences) - 1:
            # On a atteint le minimum, regarder si la prochaine phrase nous ferait dépasser max
            next_sentence_length = len(sentences[i + 1])
            if current_length + next_sentence_length + 1 > max_chunk_size:
                should_create_chunk = True

        if should_create_chunk:
            # Sauvegarder le chunk
            chunk_text = ' '.join(current_chunk)
            if chunk_text:
                chunks.append(chunk_text)

            # Préparer le prochain chunk avec overlap
            if i < len(sentences) - 1:  # Pas le dernier
                overlap_sentences = []
                overlap_length = 0

                # Prendre des phrases de la fin pour l'overlap
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

    # Traiter le dernier chunk s'il reste du contenu
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        # Si le dernier chunk est trop petit et on a déjà des chunks, le fusionner avec le précédent
        if len(chunk_text) < min_chunk_size and len(chunks) > 0:
            chunks[-1] = chunks[-1] + ' ' + chunk_text
        elif chunk_text:
            chunks.append(chunk_text)

    return chunks


class TopicExtractor:
    """
    Classe pour extraire des thèmes à partir de textes en utilisant BERTopic.
    """

    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        topic_keywords_initial: int = 20,
        topic_keywords_final: int = 5,
        logger: Optional[logging.Logger] = None,
        llm_endpoint: str = "http://localhost:11434/api/generate",
        llm_model: str = "mistral:latest",
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 20,
        llm_timeout: int = 120
    ):
        self.embedding_model = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')
        self.topic_keywords_initial = topic_keywords_initial
        self.topic_keywords_final = topic_keywords_final
        self.logger = logger or logging.getLogger(__name__)
        self.stemmer = SnowballStemmer("english")
        self.topic_model = None

        # --- Ollama / Mistral config ---
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
        Construit un modèle de topics en utilisant BERTopic.

        Args:
            chunk_texts: Liste de textes à analyser
            min_topic_size: Taille minimale d'un topic
            nr_topics: Nombre de topics (None = auto)

        Returns:
            Tuple de (topic_model, topics, probabilities)
            - topics: Liste d'IDs de topics pour chaque chunk (-1 = outlier)
            - probabilities: Array de shape (n_chunks, n_topics)
        """
        start_time = time.time()
        self.logger.info("Training BERTopic model...")
        self.logger.info(f"Computing embeddings for {len(chunk_texts)} chunks...")

        # FIX #1: Adaptive min_cluster_size to prevent duplicate topics
        # For large datasets, use ~1-2% as minimum (prevents over-fragmentation)
        if min_topic_size is None:
            min_topic_size = max(10, len(chunk_texts) // 100)
        self.logger.info(f"Using adaptive min_cluster_size={min_topic_size} for {len(chunk_texts)} chunks")

        # Créer le vectorizer avec stop words
        # min_df doit être adapté au nombre de documents
        min_df_value = min(2, max(1, len(chunk_texts) // 5))  # Au moins 1, au plus 2

        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=min_df_value,
            ngram_range=(1, 3),  # Inclure les trigrammes
            max_features=1000     # Plus de features
        )

        # UMAP personnalisé avec verbose
        # CRITICAL: n_components must be STRICTLY LESS than n_samples (not equal)
        # Also n_neighbors must be < n_samples
        n_samples = len(chunk_texts)

        # Ensure n_components < n_samples (not <=)
        max_components = max(2, n_samples - 2)  # Leave at least 2 samples margin
        n_components = min(5, max_components)

        # Ensure n_neighbors < n_samples
        max_neighbors = max(2, n_samples - 2)
        n_neighbors_umap = min(15, max_neighbors)

        self.logger.info(f"UMAP config: n_samples={n_samples}, n_components={n_components}, n_neighbors={n_neighbors_umap}")

        umap_model = UMAP(
            n_neighbors=n_neighbors_umap,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            verbose=True  # Afficher la progression UMAP
        )

        # Modèle HDBSCAN standard
        # Note: prediction_data=True requis pour calculate_probabilities (multi-topic assignment)
        # Optimisé pour grands datasets avec core_dist_n_jobs=1
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,  # Nécessaire pour probabilités multi-topics
            core_dist_n_jobs=1  # Réduit pics mémoire (séquentiel au lieu de parallèle)
        )

        # Initialiser BERTopic avec composants personnalisés
        # Note: calculate_probabilities=True pour permettre l'assignation multi-topics
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=nr_topics,
            calculate_probabilities=True,  # Active les probabilités pour chaque chunk-topic
            verbose=True  # Mode verbose pour la progression
        )

        # Fit et transform avec logging de progression
        self.logger.info(f"Starting BERTopic pipeline on {len(chunk_texts)} text chunks...")
        self.logger.info("  → Stage 1/4: Computing embeddings (GPU-accelerated)...")
        self.logger.info("  → Stage 2/4: UMAP dimensionality reduction (kNN-based)...")
        self.logger.info("  → Stage 3/4: HDBSCAN clustering (CPU, typically 30s-2min)...")

        # Wrapper pour afficher un message après HDBSCAN
        original_extract_topics = topic_model._extract_topics

        def extract_topics_with_logging(*args, **kwargs):
            self.logger.info("  → Stage 4/5: Computing c-TF-IDF representations (CPU-intensive, may take 1-5min)...")
            self.logger.info("              This step is working even if GPU usage is 0% - please be patient...")
            return original_extract_topics(*args, **kwargs)

        topic_model._extract_topics = extract_topics_with_logging

        # FIT & TRANSFORM - C'est ici que la magie opère
        topics, probabilities = topic_model.fit_transform(chunk_texts)

        # Restaurer la méthode originale pour permettre le pickle
        topic_model._extract_topics = original_extract_topics

        self.logger.info("  → Stage 5/5: Topic probabilities computed!")
        self.logger.info("✓ BERTopic pipeline completed!")

        n_topics = len(set(topics)) - 1  # excluding outliers
        elapsed = time.time() - start_time
        self.logger.info(f"✓ Training BERTopic model ({n_topics} topics found) - {elapsed:.2f}s for {len(chunk_texts)} chunks")

        self.topic_model = topic_model
        return topic_model, topics, probabilities

    def _process_topic_keywords(
        self,
        topic_words: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Traite les keywords d'un topic avec stemming et déduplication des n-grams.

        Args:
            topic_words: Liste de (word, score) de BERTopic

        Returns:
            Liste de (word, score) tuples, dédupliqués et stemmés
        """
        if not topic_words:
            return []

        # Étape 1: Stemmer et fusionner les variantes (prendre top N initialement)
        stem_scores = {}

        for word, score in topic_words[:self.topic_keywords_initial]:
            # Stemmer avec NLTK SnowballStemmer
            # Pour les phrases multi-mots, stemmer chaque mot individuellement
            words = word.split()
            stemmed_words = [self.stemmer.stem(w.lower()) for w in words]
            stem = " ".join(stemmed_words)

            # Fusionner les scores pour le même stem (somme des scores)
            if stem not in stem_scores:
                stem_scores[stem] = 0
            stem_scores[stem] += score

        # Étape 2: Filtrer les n-grams redondants
        # Stratégie: Garder les n-grams longs, retirer les courts qui sont des sous-chaînes
        # Trier par longueur (descendant) puis score (descendant)
        sorted_items = sorted(
            stem_scores.items(),
            key=lambda x: (len(x[0].split()), x[1]),  # Trier par nombre de mots, puis score
            reverse=True
        )

        filtered = []
        for stem, score in sorted_items:
            is_redundant = False

            # Vérifier si ce stem est une sous-chaîne d'un stem déjà gardé
            for kept_stem, kept_score in filtered:
                # Vérifier si le stem actuel est contenu dans un stem gardé
                # Utiliser des checks de word boundary pour éviter les faux positifs
                stem_words = set(stem.split())
                kept_words = set(kept_stem.split())

                # Si tous les mots du stem actuel sont dans le stem gardé, c'est redondant
                if stem_words.issubset(kept_words):
                    is_redundant = True
                    break

            if not is_redundant:
                # Avant d'ajouter, retirer les stems gardés qui sont des sous-chaînes du stem actuel
                to_remove = []
                for kept_stem, kept_score in filtered:
                    kept_words = set(kept_stem.split())
                    stem_words = set(stem.split())

                    # Si le stem gardé est un subset du stem actuel, marquer pour suppression
                    if kept_words.issubset(stem_words) and kept_stem != stem:
                        to_remove.append((kept_stem, kept_score))

                # Retirer les items redondants gardés
                for item in to_remove:
                    filtered.remove(item)

                filtered.append((stem, score))

        # Étape 3: Re-trier par score et prendre top N
        filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
        final_count = min(max(len(filtered_sorted), 3), self.topic_keywords_final)
        top_keywords = filtered_sorted[:final_count]

        # Retourner avec les formes stemmées
        result = [(stem, score) for stem, score in top_keywords]

        self.logger.debug(f"Processed keywords: {len(topic_words)} -> {len(result)} (removed {len(topic_words) - len(result)} redundant)")

        return result

    def get_topic_label(
        self,
        topic_words: List[Tuple[str, float]],
        use_llm: bool = False
    ) -> str:
        """
        Génère un label pour un topic à partir de ses keywords.

        Args:
            topic_words: Liste de (word, score) tuples de BERTopic
            use_llm: Si True, utilise un LLM pour générer un label lisible

        Returns:
            Label du topic (chaîne de caractères)
        """
        # Traiter les keywords: stemmer et supprimer les n-grams redondants
        processed_keywords = self._process_topic_keywords(topic_words)

        if use_llm:
            # FIX #2: Use LLM to create readable labels
            return self._label_topic_with_llm(processed_keywords)
        else:
            # Fallback: Concaténer les top 3 mots (stemmed - may be unreadable)
            top_words = [word for word, score in processed_keywords]
            label = " ".join(top_words[:3])
            self.logger.debug(f"Keyword-based label: {label}")
            return label

    def _label_topic_with_llm(
        self,
        keywords: List[Tuple[str, float]],
    ) -> str:
        """
        Utilise Mistral via Ollama (API /api/generate) pour générer un label lisible.
        """
        try:
            import requests

            # keywords déjà stemmés: on peut les donner tels quels,
            # ou nettoyer un peu pour aider le modèle.
            keyword_list = ", ".join([f"{word} ({score:.3f})" for word, score in keywords[:10]])

            prompt = f"""You are a helpful assistant that creates concise, readable topic labels.

    Given the following keywords and importance scores from topic modeling, generate a short topic label (2-5 words max).

    Keywords: {keyword_list}

    Rules:
    - Return ONLY the label, nothing else
    - Proper English (no stemmed fragments like "effici", "accuraci")
    - No quotes

    Label:"""

            resp = requests.post(
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

            if resp.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            label = (data.get("response") or "").strip()

            # Nettoyage simple
            label = label.replace("\n", " ").strip()
            label = label.strip('"').strip("'")

            # Garde-fous
            if not label:
                raise RuntimeError("Empty label returned by Ollama")

            # Limiter à 5 mots max, au cas où
            words = label.split()
            if len(words) > 5:
                label = " ".join(words[:5])

            return label

        except Exception as e:
            self.logger.warning(f"LLM labeling failed (Ollama/Mistral): {e}, falling back to keyword concatenation")
            top_words = [word for word, score in keywords[:3]]
            return " ".join(top_words)


    def get_all_topics_info(self, use_llm_labels: bool = True) -> dict:
        """
        Récupère les informations sur tous les topics découverts.

        Args:
            use_llm_labels: Si True, utilise un LLM pour générer des labels lisibles

        Returns:
            Dictionnaire contenant les infos de tous les topics
        """
        if self.topic_model is None:
            raise ValueError("Topic model not trained yet. Call build_topic_model() first.")

        topics_info = {}
        topic_freq = self.topic_model.get_topic_freq()

        self.logger.info(f"Generating labels for {len(topic_freq)-1} topics (use_llm={use_llm_labels})...")

        for topic_id in topic_freq['Topic']:
            if topic_id == -1:  # Skip outliers
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
                'count': count
            }

        return topics_info

    def assign_topics_to_chunks(
        self,
        topics: List[int],
        probabilities: np.ndarray,
        top_k: int = 3,
        min_prob: float = 0.01
    ) -> List[List[Tuple[int, float]]]:
        """
        Assigne les top-K topics à chaque chunk avec leurs probabilités.

        Args:
            topics: Liste d'IDs de topics (sortie de fit_transform)
            probabilities: Array de probabilités (n_chunks, n_topics)
            top_k: Nombre de topics à assigner par chunk
            min_prob: Probabilité minimale pour assigner un topic

        Returns:
            Liste de listes de (topic_id, probability) pour chaque chunk
        """
        assignments = []

        for idx in range(len(topics)):
            # Skip outliers (topic -1)
            if topics[idx] == -1:
                assignments.append([])
                continue

            # Multi-topic assignment avec probabilités
            if probabilities is not None and probabilities.ndim > 1:
                chunk_probs = probabilities[idx]

                # Stratégie: Utiliser TOP-K topics (avec leurs probabilités comme poids)
                # Cela garantit que chaque chunk a plusieurs topics pour les CO_PRESENCE edges

                # Obtenir les indices des top-K topics triés par probabilité
                top_k_indices = np.argsort(chunk_probs)[-top_k:][::-1]

                chunk_assignments = []
                for topic_id in top_k_indices:
                    prob = chunk_probs[topic_id]

                    # Skip si topic est -1 (outlier) ou probabilité trop faible (< min_prob)
                    if topic_id == -1 or prob < min_prob:
                        continue

                    chunk_assignments.append((int(topic_id), float(prob)))

                assignments.append(chunk_assignments)
            else:
                # Fallback: Hard assignment si pas de probabilités
                assignments.append([(int(topics[idx]), 1.0)])

        return assignments

    def generate_html_report(
        self,
        chunk_texts: List[str],
        topics: List[int],
        probabilities: np.ndarray,
        output_dir: str = "topic_extraction_results",
        sample_chunks_indices: Optional[List[int]] = None,
        chunk_ids: Optional[List[str]] = None
    ) -> str:
        """
        Génère un rapport HTML récapitulatif des topics extraits.

        Args:
            chunk_texts: Liste des textes analysés
            topics: Liste d'IDs de topics (sortie de fit_transform)
            probabilities: Array de probabilités (n_chunks, n_topics)
            output_dir: Dossier de sortie pour le rapport
            sample_chunks_indices: Indices des chunks à détailler (3 chunks aléatoires si None)
            chunk_ids: Liste optionnelle des IDs de chunks (format: "chunk::doc_id::chunk_idx")

        Returns:
            Chemin du fichier HTML généré
        """
        # Générer les chunk IDs si non fournis (pour compatibilité PostgreSQL)
        if chunk_ids is None:
            chunk_ids = [f"chunk::{idx}::0" for idx in range(len(chunk_texts))]
        # Créer le dossier de sortie
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Timestamp pour le nom de fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sélectionner 3 chunks aléatoires si non spécifiés
        if sample_chunks_indices is None:
            import random
            random.seed(42)  # Pour la reproductibilité
            sample_chunks_indices = random.sample(range(len(chunk_texts)), min(3, len(chunk_texts)))

        # Récupérer les infos des topics avec labels LLM
        topics_info = self.get_all_topics_info(use_llm_labels=True)
        assignments = self.assign_topics_to_chunks(topics, probabilities, top_k=3)

        # Construire le HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topic Extraction Report - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
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
        .keyword-score {{
            color: #7f8c8d;
            font-size: 0.85em;
        }}
        .chunks-section {{
            margin-top: 30px;
        }}
        .chunk-card {{
            background-color: white;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chunk-header {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .chunk-text {{
            color: #34495e;
            line-height: 1.6;
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .chunk-topics {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .topic-assignment {{
            background-color: #e8f5e9;
            padding: 8px 12px;
            border-radius: 5px;
            border-left: 3px solid #27ae60;
            flex: 1;
            min-width: 200px;
        }}
        .topic-assignment-name {{
            font-weight: bold;
            color: #27ae60;
            font-size: 0.95em;
        }}
        .topic-assignment-prob {{
            color: #7f8c8d;
            font-size: 0.85em;
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
            background-color: #27ae60;
            transition: width 0.3s ease;
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
    <h1>🔍 Topic Extraction Report</h1>

    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="summary-stats">
            <div class="stat-box">
                <div class="stat-label">Total Chunks Analyzed</div>
                <div class="stat-value">{len(chunk_texts)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Topics Discovered</div>
                <div class="stat-value">{len(topics_info)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg Topics per Chunk</div>
                <div class="stat-value">{sum(len(a) for a in assignments) / len(assignments):.2f}</div>
            </div>
        </div>
    </div>

    <h2>📊 Discovered Topics</h2>
    <div class="topics-grid">
"""

        # Ajouter chaque topic
        for topic_id, info in sorted(topics_info.items(), key=lambda x: x[1]['count'], reverse=True):
            html += f"""
        <div class="topic-card">
            <div class="topic-header">
                <div class="topic-id">Topic {topic_id}</div>
                <div class="topic-count">{info['count']} docs</div>
            </div>
            <div class="topic-label">"{info['label']}"</div>
            <div class="keywords">
"""
            for keyword, score in info['keyword_scores']:
                html += f'                <span class="keyword">{keyword} <span class="keyword-score">({score:.3f})</span></span>\n'

            html += """            </div>
        </div>
"""

        html += """    </div>

    <h2>📝 Chunk-Topic Assignments</h2>
    <div class="chunks-section">
"""

        # Ajouter les assignations pour chaque chunk
        for idx, (text, chunk_topics) in enumerate(zip(chunk_texts, assignments)):
            chunk_id = chunk_ids[idx] if idx < len(chunk_ids) else f"chunk::{idx}::0"
            html += f"""
        <div class="chunk-card">
            <div class="chunk-header">Chunk {idx + 1} <span style="color: #7f8c8d; font-weight: normal; font-size: 0.9em;">(ID: {chunk_id})</span></div>
            <div class="chunk-text">{text}</div>
            <div class="chunk-topics">
"""
            for topic_id, prob in chunk_topics:
                topic_label = topics_info.get(topic_id, {}).get('label', 'unknown')
                prob_percent = prob * 100
                html += f"""
                <div class="topic-assignment">
                    <div class="topic-assignment-name">Topic {topic_id}: {topic_label}</div>
                    <div class="topic-assignment-prob">Probability: {prob:.3f} ({prob_percent:.1f}%)</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {prob_percent}%"></div>
                    </div>
                </div>
"""

            html += """            </div>
        </div>
"""

        html += """    </div>

    <h2>🔬 Detailed Analysis: Sample Chunks</h2>
    <p>Deep dive into 3 randomly selected chunks with their assigned topics and complete topic details.</p>
    <div class="chunks-section">
"""

        # Ajouter la section détaillée pour les 3 chunks échantillons
        for sample_idx in sample_chunks_indices:
            chunk_text = chunk_texts[sample_idx]
            chunk_topics = assignments[sample_idx]
            chunk_id = chunk_ids[sample_idx] if sample_idx < len(chunk_ids) else f"chunk::{sample_idx}::0"

            html += f"""
        <div class="chunk-card" style="border-left: 5px solid #e74c3c;">
            <div class="chunk-header" style="font-size: 1.2em;">📄 Sample Chunk #{sample_idx + 1} <span style="color: #7f8c8d; font-weight: normal; font-size: 0.85em;">(ID: {chunk_id})</span></div>
            <div class="chunk-text" style="font-size: 1.05em;">{chunk_text}</div>

            <h3 style="color: #e74c3c; margin-top: 20px; margin-bottom: 15px;">Assigned Topics with Full Details:</h3>
"""

            # Pour chaque topic assigné à ce chunk, afficher les détails complets
            for topic_id, prob in chunk_topics:
                topic_info = topics_info.get(topic_id, {})
                topic_label = topic_info.get('label', 'unknown')
                keywords = topic_info.get('keywords', [])
                keyword_scores = topic_info.get('keyword_scores', [])
                raw_keywords = topic_info.get('raw_keywords', [])
                count = topic_info.get('count', 0)
                prob_percent = prob * 100

                html += f"""
            <div style="background-color: #fff3cd; padding: 20px; margin-bottom: 20px; border-radius: 8px; border-left: 5px solid #ffc107;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="font-size: 1.2em; font-weight: bold; color: #2c3e50;">Topic {topic_id}: "{topic_label}"</div>
                    <div style="background-color: #ffc107; color: white; padding: 5px 15px; border-radius: 15px; font-weight: bold;">
                        {prob:.3f} ({prob_percent:.1f}%)
                    </div>
                </div>

                <div style="margin-top: 15px;">
                    <div style="font-weight: bold; color: #7f8c8d; margin-bottom: 8px;">📊 Topic Statistics:</div>
                    <div style="margin-left: 15px; color: #34495e;">
                        • Total documents in this topic: <strong>{count}</strong><br>
                        • Assignment probability for this chunk: <strong>{prob:.4f}</strong>
                    </div>
                </div>

                <div style="margin-top: 15px;">
                    <div style="font-weight: bold; color: #7f8c8d; margin-bottom: 8px;">🔑 Processed Keywords (stemmed & deduplicated):</div>
                    <div style="margin-left: 15px;">
"""
                for kw in keywords:
                    html += f'                        <span style="background-color: #e8f5e9; padding: 5px 12px; margin: 3px; border-radius: 5px; display: inline-block; font-weight: 600; color: #27ae60;">{kw}</span>\n'

                html += """                    </div>
                </div>

                <div style="margin-top: 15px;">
                    <div style="font-weight: bold; color: #7f8c8d; margin-bottom: 8px;">📈 Keyword Scores (with weights):</div>
                    <div style="margin-left: 15px;">
"""
                for kw, score in keyword_scores:
                    score_width = min(score * 100, 100)
                    html += f"""
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                                <span style="font-weight: 600; color: #2c3e50;">{kw}</span>
                                <span style="color: #7f8c8d; font-size: 0.9em;">{score:.4f}</span>
                            </div>
                            <div style="height: 8px; background-color: #ecf0f1; border-radius: 4px; overflow: hidden;">
                                <div style="height: 100%; background: linear-gradient(90deg, #27ae60, #2ecc71); width: {score_width}%;"></div>
                            </div>
                        </div>
"""

                html += """                    </div>
                </div>

                <div style="margin-top: 15px;">
                    <div style="font-weight: bold; color: #7f8c8d; margin-bottom: 8px;">🔤 Raw BERTopic Keywords (top 10):</div>
                    <div style="margin-left: 15px; font-size: 0.9em; color: #555;">
"""
                for raw_kw, raw_score in raw_keywords[:10]:
                    html += f'                        <span style="background-color: #f8f9fa; padding: 3px 8px; margin: 2px; border-radius: 3px; display: inline-block; border: 1px solid #dee2e6;">{raw_kw} <span style="color: #7f8c8d;">({raw_score:.3f})</span></span>\n'

                html += """                    </div>
                </div>
            </div>
"""

            html += """        </div>
"""

        html += f"""
    </div>

    <div class="timestamp">
        Generated on {timestamp}<br>
        Sample chunks: {', '.join(f'#{i+1}' for i in sample_chunks_indices)}
    </div>
</body>
</html>
"""

        # Sauvegarder le fichier HTML
        output_file = output_path / f"topic_extraction_report_{timestamp}.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        return str(output_file)


def main():
    """
    Exemple d'utilisation de TopicExtractor.
    """
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Templates de documents pour générer 100 documents variés
    templates = [
        """Climate change is affecting global temperatures and weather patterns across the world.
        Scientists have observed unprecedented changes in ice caps and glaciers. The melting of polar ice
        is accelerating at rates far exceeding previous predictions. Rising sea levels threaten coastal
        communities worldwide. Extreme weather events are becoming more frequent and severe. Governments
        are implementing new policies to reduce carbon emissions. Renewable energy sources like solar and
        wind power are becoming increasingly viable alternatives to fossil fuels.""",

        """The stock market experienced significant volatility today as investors reacted to economic data.
        Technology companies led the market rally with strong quarterly earnings. Major indices showed mixed
        performance throughout the trading session. Analysts are concerned about the economic outlook for the
        coming quarters. Interest rate decisions by central banks continue to influence market sentiment.
        Retail investors are increasingly participating in equity markets through mobile trading platforms.
        Financial advisors recommend diversification to manage portfolio risk.""",

        """Artificial intelligence and machine learning are transforming various industries at an unprecedented pace.
        Deep learning models are achieving state-of-the-art results in computer vision and natural language processing.
        Companies are investing heavily in AI research and development. Ethical considerations around AI deployment
        are gaining importance. Automation is reshaping the future of work across sectors. Neural networks continue
        to improve in accuracy and efficiency. AI applications in healthcare are showing promising results in
        disease diagnosis and drug discovery.""",

        """The recent election results surprised many political analysts and pollsters. Voter turnout reached
        historic levels in several key districts. Campaign strategies focused heavily on social media engagement.
        Political polarization continues to shape electoral outcomes. Young voters played a crucial role in
        determining election results. Debates around election integrity and voting access remain contentious.
        The newly elected officials face significant challenges in addressing public concerns.""",

        """Global supply chain disruptions are affecting businesses worldwide. Manufacturing delays are causing
        shortages of consumer goods. Shipping costs have increased dramatically due to logistics challenges.
        Companies are reconsidering their supply chain strategies to improve resilience. Just-in-time inventory
        systems are being reevaluated in light of recent disruptions. International trade tensions add complexity
        to global commerce. Digital transformation is helping companies adapt to supply chain challenges.""",

        """Space exploration is entering a new era with increased private sector involvement. Commercial spaceflight
        companies are making space more accessible. Mars missions are being planned by multiple space agencies.
        Satellite technology is revolutionizing communications and Earth observation. The International Space Station
        continues to serve as a platform for scientific research. Space tourism is becoming a reality for wealthy
        individuals. Debates about space regulation and resource extraction are intensifying.""",

        """Healthcare systems worldwide are facing unprecedented challenges and opportunities. Medical research is
        advancing rapidly with breakthrough treatments. Telemedicine is expanding access to healthcare services.
        Pharmaceutical companies are developing new medications. Patient care is improving through technology.
        Healthcare workers are dealing with increased workload demands. Public health policies are being reformed.""",

        """Education systems are adapting to digital transformation. Online learning platforms are expanding globally.
        Students are engaging with new educational technologies. Teachers are developing innovative teaching methods.
        Educational institutions are rethinking traditional approaches. Access to education is improving worldwide.
        Lifelong learning is becoming increasingly important."""
    ]

    # Générer 1000 documents en répétant et variant les templates
    sample_documents = []
    import random
    random.seed(42)  # Pour reproductibilité

    for i in range(1000):
        # Sélectionner un template aléatoire
        template = random.choice(templates)
        # Ajouter une petite variation en ajoutant un numéro
        doc = template + f" This is document number {i+1} with specific context."
        sample_documents.append(doc)

    # Utiliser la nouvelle fonction de chunking pour créer des chunks de taille appropriée
    logger.info("Chunking documents with minimum size requirement...")
    all_chunks = []
    chunk_ids = []

    for doc_idx, doc in enumerate(sample_documents):
        chunks = chunk_text_for_topics(doc, min_chunk_size=400, max_chunk_size=800, overlap=100)

        # Créer les chunk IDs au format PostgreSQL: "chunk::doc_id::chunk_idx"
        for chunk_idx, chunk_text in enumerate(chunks):
            all_chunks.append(chunk_text)
            # doc_idx + 1 car les documents sont numérotés de 1 à 1000
            chunk_ids.append(f"chunk::{doc_idx + 1}::{chunk_idx}")

    logger.info(f"Created {len(all_chunks)} chunks from {len(sample_documents)} documents")

    # Afficher quelques statistiques sur les chunks
    chunk_lengths = [len(chunk) for chunk in all_chunks]
    logger.info(f"Chunk size stats: min={min(chunk_lengths)}, max={max(chunk_lengths)}, avg={sum(chunk_lengths)/len(chunk_lengths):.0f}")

    sample_texts = all_chunks

    # Initialiser l'extracteur
    logger.info("Initializing TopicExtractor...")
    extractor = TopicExtractor(logger=logger)

    # Entraîner le modèle
    logger.info("\nTraining topic model...")
    topic_model, topics, probabilities = extractor.build_topic_model(
        sample_texts,
        min_topic_size=None,  # Auto-adaptive (will use len//100, min 10) - FIX #1
        nr_topics=None        # Auto-detect
    )

    # Afficher les topics découverts
    logger.info("\n" + "="*80)
    logger.info("TOPICS DÉCOUVERTS")
    logger.info("="*80)

    # FIX #2: Use LLM to generate readable topic labels
    topics_info = extractor.get_all_topics_info(use_llm_labels=True)
    for topic_id, info in topics_info.items():
        logger.info(f"\nTopic {topic_id}: {info['label']}")
        logger.info(f"  Count: {info['count']} documents")
        logger.info(f"  Keywords: {', '.join(info['keywords'])}")
        logger.info(f"  Keyword scores: {info['keyword_scores']}")

    # Assigner les topics aux chunks
    logger.info("\n" + "="*80)
    logger.info("TOPIC ASSIGNMENTS (Top-3 per chunk)")
    logger.info("="*80)

    assignments = extractor.assign_topics_to_chunks(topics, probabilities, top_k=3)
    for idx, chunk_topics in enumerate(assignments):
        logger.info(f"\nChunk {idx}: {sample_texts[idx][:60]}...")
        for topic_id, prob in chunk_topics:
            topic_label = topics_info.get(topic_id, {}).get('label', 'unknown')
            logger.info(f"  → Topic {topic_id} ({topic_label}): {prob:.3f}")

    # Générer le rapport HTML
    logger.info("\n" + "="*80)
    logger.info("GENERATING HTML REPORT")
    logger.info("="*80)

    # Fixer 3 chunks spécifiques pour l'analyse détaillée (toujours les mêmes)
    # On choisit les chunks 10, 50, et 90 pour avoir une diversité
    fixed_sample_chunks = [10, 50, 90] if len(sample_texts) > 90 else [0, min(len(sample_texts)//2, len(sample_texts)-1), len(sample_texts)-1]

    logger.info(f"\nFixed sample chunks for detailed analysis: {fixed_sample_chunks}")

    output_file = extractor.generate_html_report(
        chunk_texts=sample_texts,
        topics=topics,
        probabilities=probabilities,
        output_dir="topic_extraction_results",
        sample_chunks_indices=fixed_sample_chunks,  # Utiliser les chunks fixés
        chunk_ids=chunk_ids  # Passer les chunk IDs au format PostgreSQL
    )

    logger.info(f"\n✓ HTML report saved to: {output_file}")
    logger.info(f"  Open this file in your browser to view the detailed report.")


if __name__ == "__main__":
    main()
