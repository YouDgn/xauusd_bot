# ============================================================
# news_sentiment.py — Module News & Analyse de Sentiment
# ============================================================

import feedparser
import requests
import threading
import time
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from textblob import TextBlob
import numpy as np
import config

logger = logging.getLogger(__name__)


class NewsItem:
    """Représente un article de news parsé."""
    def __init__(self, title: str, summary: str, published: datetime, source: str):
        self.title     = title
        self.summary   = summary
        self.published = published
        self.source    = source
        self.sentiment_score: float = 0.0
        self.relevance_score: float = 0.0

    def __repr__(self):
        return (f"[{self.source}] {self.published:%H:%M} | "
                f"Sent={self.sentiment_score:+.2f} | {self.title[:60]}...")


class SentimentAnalyzer:
    """
    Analyse de sentiment combinant TextBlob et règles métier pour le trading de l'Or.
    Peut être upgradé vers un modèle HuggingFace FinBERT.
    """

    # Mots bullish pour l'or (gold monte)
    BULLISH_GOLD_WORDS = {
        "crisis", "war", "conflict", "geopolitical", "inflation", "recession",
        "safe haven", "uncertainty", "fear", "rate cut", "dovish", "weak dollar",
        "deficit", "debt ceiling", "bank failure", "panic", "crash", "risk off",
        "gold rally", "buying gold", "gold surge", "gold rises", "gold climbs",
        "central bank buying", "sanctions", "negative real rates"
    }

    # Mots bearish pour l'or (gold baisse)
    BEARISH_GOLD_WORDS = {
        "rate hike", "hawkish", "strong dollar", "risk on", "recovery",
        "growth", "bull market", "gold falls", "gold drops", "gold decline",
        "gold sell", "profit taking", "dollar rally", "yields rise",
        "tightening", "fed hike", "inflation easing", "soft landing"
    }

    def __init__(self):
        self._try_load_finbert()

    def _try_load_finbert(self):
        """Tente de charger FinBERT pour une analyse plus précise."""
        self.finbert = None
        self.finbert_tokenizer = None
        try:
            from transformers import pipeline
            logger.info("Chargement de FinBERT pour l'analyse de sentiment...")
            self.finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # CPU (évite les conflits avec DirectML)
            )
            logger.info("✅ FinBERT chargé avec succès.")
        except Exception as e:
            logger.warning(f"FinBERT non disponible ({e}), utilisation de TextBlob.")

    def analyze(self, text: str) -> float:
        """
        Retourne un score de sentiment entre -1.0 (très bearish) et +1.0 (très bullish).
        Combine TextBlob + règles spécifiques à l'or.
        """
        text_lower = text.lower()

        # 1) Score TextBlob (général)
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity  # [-1, 1]

        # 2) Score FinBERT si disponible
        if self.finbert is not None:
            try:
                # FinBERT attend max 512 tokens
                truncated = text[:512]
                result = self.finbert(truncated)[0]
                label = result["label"].lower()
                score = result["score"]
                if label == "positive":
                    finbert_score = score
                elif label == "negative":
                    finbert_score = -score
                else:
                    finbert_score = 0.0
            except Exception:
                finbert_score = textblob_score
        else:
            finbert_score = textblob_score

        # 3) Score spécifique Or (règles métier)
        gold_score = 0.0
        for word in self.BULLISH_GOLD_WORDS:
            if word in text_lower:
                gold_score += 0.15
        for word in self.BEARISH_GOLD_WORDS:
            if word in text_lower:
                gold_score -= 0.15

        # Clipper le score or entre -1 et 1
        gold_score = max(-1.0, min(1.0, gold_score))

        # 4) Pondération finale
        if self.finbert is not None:
            final = 0.40 * finbert_score + 0.35 * gold_score + 0.25 * textblob_score
        else:
            final = 0.50 * gold_score + 0.50 * textblob_score

        return max(-1.0, min(1.0, final))

    def compute_relevance(self, text: str) -> float:
        """Calcule un score de pertinence [0, 1] pour l'or."""
        text_lower = text.lower()
        score = 0.0
        for keyword in config.GOLD_KEYWORDS:
            if keyword in text_lower:
                score += 1.0 / len(config.GOLD_KEYWORDS)
        return min(1.0, score * 3)  # Amplifier pour les articles très pertinents


class NewsSentimentModule:
    """
    Module complet : scraping RSS + scoring sentiment + agrégation.
    Tourne dans un thread séparé pour ne pas bloquer le bot.
    """

    def __init__(self):
        self.analyzer         = SentimentAnalyzer()
        self.news_items:  List[NewsItem] = []
        self.current_score:   float = 0.0   # Score agrégé [-1, 1]
        self.last_update:     Optional[datetime] = None
        self._lock            = threading.Lock()
        self._running         = False
        self._thread: Optional[threading.Thread] = None

    # ── API Publique ───────────────────────────────────────────

    def start(self):
        """Lance le thread de scraping en arrière-plan."""
        self._running = True
        self._thread  = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        logger.info("📰 Module News démarré (thread arrière-plan).")

    def stop(self):
        """Arrête le thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("📰 Module News arrêté.")

    def get_current_sentiment(self) -> Tuple[float, List[NewsItem]]:
        """
        Retourne (score_agrégé, liste_articles_récents).
        Thread-safe.
        """
        with self._lock:
            return self.current_score, list(self.news_items[:5])

    def force_update(self):
        """Force une mise à jour immédiate (bloquant)."""
        self._fetch_and_score()

    # ── Thread Interne ─────────────────────────────────────────

    def _update_loop(self):
        """Boucle principale du thread news."""
        while self._running:
            try:
                self._fetch_and_score()
            except Exception as e:
                logger.error(f"Erreur module news : {e}")
            time.sleep(config.NEWS_UPDATE_INTERVAL)

    def _fetch_and_score(self):
        """Scrape tous les feeds RSS et calcule le score agrégé."""
        all_items: List[NewsItem] = []

        for feed_url in config.RSS_FEEDS:
            items = self._parse_rss_feed(feed_url)
            all_items.extend(items)

        if not all_items:
            logger.warning("Aucun article news récupéré.")
            return

        # Filtrer les articles des dernières 6 heures
        cutoff = datetime.utcnow() - timedelta(hours=6)
        recent = [i for i in all_items if i.published >= cutoff]

        if not recent:
            recent = all_items[:20]  # Fallback : 20 derniers articles

        # Scorer chaque article
        scored_items = []
        for item in recent:
            full_text = f"{item.title} {item.summary}"
            item.sentiment_score = self.analyzer.analyze(full_text)
            item.relevance_score = self.analyzer.compute_relevance(full_text)
            if item.relevance_score > 0.05:  # Ignorer les articles hors-sujet
                scored_items.append(item)

        if not scored_items:
            logger.debug("Aucun article pertinent pour l'or.")
            return

        # Agréger avec pondération par pertinence et récence
        weighted_scores = []
        weights = []
        now = datetime.utcnow()

        for item in scored_items:
            age_hours  = (now - item.published).total_seconds() / 3600
            time_decay = np.exp(-age_hours / 3)  # Décroissance exponentielle
            weight     = item.relevance_score * time_decay
            weighted_scores.append(item.sentiment_score * weight)
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight > 0:
            aggregated = sum(weighted_scores) / total_weight
        else:
            aggregated = 0.0

        with self._lock:
            self.news_items   = sorted(scored_items, key=lambda x: x.published, reverse=True)
            self.current_score = round(max(-1.0, min(1.0, aggregated)), 4)
            self.last_update   = datetime.utcnow()

        logger.info(
            f"📰 News mise à jour | {len(scored_items)} articles pertinents | "
            f"Score sentiment: {self.current_score:+.3f}"
        )

    def _parse_rss_feed(self, url: str) -> List[NewsItem]:
        """Parse un feed RSS et retourne les NewsItems."""
        items = []
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; GoldBot/1.0)"}
            response = requests.get(url, headers=headers, timeout=10)
            feed = feedparser.parse(response.content)

            for entry in feed.entries[:30]:  # Max 30 par feed
                title   = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                # Nettoyer le HTML
                summary = BeautifulSoup(summary, "html.parser").get_text()

                # Parser la date
                published = datetime.utcnow()
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        import calendar
                        published = datetime(*entry.published_parsed[:6])
                    except Exception:
                        pass

                source = feed.feed.get("title", url.split("/")[2])
                items.append(NewsItem(title, summary[:500], published, source))

        except Exception as e:
            logger.debug(f"Erreur RSS {url}: {e}")

        return items

    def get_summary_string(self) -> str:
        """Retourne une description textuelle du sentiment."""
        score = self.current_score
        if score > 0.5:
            return f"🟢 TRÈS BULLISH ({score:+.2f})"
        elif score > 0.2:
            return f"🟡 BULLISH ({score:+.2f})"
        elif score > -0.2:
            return f"⚪ NEUTRE ({score:+.2f})"
        elif score > -0.5:
            return f"🟠 BEARISH ({score:+.2f})"
        else:
            return f"🔴 TRÈS BEARISH ({score:+.2f})"