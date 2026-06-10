"""PII detection rate metric for synthetic document evaluation.

Detects PII categories using regex patterns (stdlib only).
For production use, swap _detect_regex for a Presidio-based detector:

    from presidio_analyzer import AnalyzerEngine
    engine = AnalyzerEngine()
    results = engine.analyze(text=text, language="en")

The public API (detect_pii, pii_density, pii_leakage_score) is identical
whether using the regex or Presidio backend.
"""
from __future__ import annotations

import re
import math
from typing import Sequence

# ---------------------------------------------------------------------------
# Regex-based PII patterns (conservative — production should use Presidio)
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL":          re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', re.I),
    "PHONE":          re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b'),
    "SSN":            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "DATE":           re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b'),
    "CURRENCY":       re.compile(r'\$\s?\d[\d,]*(?:\.\d{2})?'),
    "IBAN":           re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})?\b'),
    "PERSON_TITLE":   re.compile(r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s[A-Z][a-z]+\b'),
    "ORG_SUFFIX":     re.compile(r'\b[A-Z][A-Za-z\s]+(?:Inc\.|LLC|Corp\.|Ltd\.|LLP)\b'),
    "IP_ADDRESS":     re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "MEDICAL_CODE":   re.compile(r'\b[A-Z]\d{2}(?:\.\d{1,4})?\b'),  # ICD-10 style
}

PII_CATEGORIES = list(_PATTERNS.keys())


def detect_pii(text: str) -> dict[str, list[str]]:
    """Return a dict mapping PII category → list of matched strings in *text*."""
    results: dict[str, list[str]] = {}
    for category, pattern in _PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            results[category] = [str(m) for m in matches]
    return results


def pii_density(text: str, per_words: int = 1000) -> dict[str, float]:
    """Return PII entities per *per_words* words for each category.

    A density of 0.0 means no PII detected for that category.
    Useful for comparing real vs. synthetic document PII profiles.
    """
    word_count = max(1, len(text.split()))
    scale = per_words / word_count
    detected = detect_pii(text)
    return {cat: len(detected.get(cat, [])) * scale for cat in PII_CATEGORIES}


def pii_leakage_score(
    synthetic_texts: Sequence[str],
    real_texts: Sequence[str],
    *,
    threshold_multiplier: float = 1.05,
) -> dict[str, object]:
    """Measure whether synthetic documents leak more PII than real documents.

    Computes mean PII density per category for both corpora. A category
    is flagged as "leaking" if the synthetic density exceeds the real
    density by more than *threshold_multiplier* (default: 5% excess).

    Returns a summary dict with:
      - per_category: {category: {real_density, synth_density, ratio, leaking}}
      - overall_leaking: True if any category is leaking
      - leaking_categories: list of flagged categories
      - pii_pass: True if no categories are leaking (privacy check passes)
    """
    def mean_density(texts: Sequence[str]) -> dict[str, float]:
        if not texts:
            return {cat: 0.0 for cat in PII_CATEGORIES}
        totals: dict[str, float] = {cat: 0.0 for cat in PII_CATEGORIES}
        for text in texts:
            d = pii_density(text)
            for cat in PII_CATEGORIES:
                totals[cat] += d.get(cat, 0.0)
        return {cat: totals[cat] / len(texts) for cat in PII_CATEGORIES}

    real_density = mean_density(real_texts)
    synth_density = mean_density(synthetic_texts)

    per_category: dict[str, dict[str, float | bool]] = {}
    leaking_categories: list[str] = []

    for cat in PII_CATEGORIES:
        real_d = real_density[cat]
        synth_d = synth_density[cat]
        # Only flag leakage if synthetic exceeds real by threshold_multiplier
        # (small amounts of PII in synthetic are expected; excess is the signal)
        threshold = real_d * threshold_multiplier
        leaking = synth_d > threshold and synth_d > 0.0
        ratio = synth_d / real_d if real_d > 0 else (float("inf") if synth_d > 0 else 1.0)
        per_category[cat] = {
            "real_density": round(real_d, 4),
            "synth_density": round(synth_d, 4),
            "ratio": round(min(ratio, 99.0), 4),
            "leaking": leaking,
        }
        if leaking:
            leaking_categories.append(cat)

    return {
        "per_category": per_category,
        "overall_leaking": len(leaking_categories) > 0,
        "leaking_categories": leaking_categories,
        "pii_pass": len(leaking_categories) == 0,
        "message": (
            "PASS: synthetic document PII density within acceptable range"
            if not leaking_categories
            else f"FAIL: excess PII in categories: {', '.join(leaking_categories)}"
        ),
    }


def distribution_shift(
    synthetic_texts: Sequence[str],
    real_texts: Sequence[str],
) -> float:
    """JS divergence between real and synthetic PII type distributions.

    A well-calibrated SDG tool should generate PII with the same
    categorical distribution as the real corpus (e.g., same ratio of
    dates to names to currencies). Large shift → synthetic documents
    have a structurally different PII profile.

    Returns a value in [0, 1]; 0 = identical distributions.
    """
    def pii_type_dist(texts: Sequence[str]) -> dict[str, float]:
        totals = {cat: 0.0 for cat in PII_CATEGORIES}
        for text in texts:
            detected = detect_pii(text)
            for cat, matches in detected.items():
                totals[cat] += len(matches)
        total_entities = sum(totals.values())
        if total_entities == 0:
            return {cat: 1.0 / len(PII_CATEGORIES) for cat in PII_CATEGORIES}
        return {cat: totals[cat] / total_entities for cat in PII_CATEGORIES}

    p = pii_type_dist(real_texts)
    q = pii_type_dist(synthetic_texts)

    # Jensen-Shannon divergence (symmetric, [0,1])
    keys = PII_CATEGORIES
    m = {k: 0.5 * (p[k] + q[k]) for k in keys}

    def kl(a: dict, b: dict) -> float:
        result = 0.0
        for k in keys:
            if a[k] > 0 and b[k] > 0:
                result += a[k] * math.log2(a[k] / b[k])
        return result

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
