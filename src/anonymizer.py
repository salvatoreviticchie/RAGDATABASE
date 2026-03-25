from __future__ import annotations

import re
from dataclasses import dataclass

# ── Try to load Presidio (full NLP-based PII detection) ──────────────────────
# Falls back to regex-only if spaCy model is not installed.
_PRESIDIO_AVAILABLE = False
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    _PRESIDIO_AVAILABLE = True
except Exception:
    pass  # spaCy model not downloaded — regex fallback will be used


# ── Regex fallback patterns ───────────────────────────────────────────────────
_PATTERNS: list[tuple[str, str, str]] = [
    # (entity_type, regex, placeholder_prefix)
    ("EMAIL",       r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",         "EMAIL"),
    ("PHONE",       r"\b(?:\+?\d[\d\s\-().]{6,}\d)\b",                                  "PHONE"),
    ("CREDIT_CARD", r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",                 "CARD"),
    ("IP_ADDRESS",  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                                     "IP"),
    ("SSN",         r"\b\d{3}-\d{2}-\d{4}\b",                                           "SSN"),
    ("IBAN",        r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",                                "IBAN"),
]


@dataclass
class PIIEntity:
    entity_type: str   # e.g. "EMAIL", "PERSON", "PHONE"
    original: str      # the original text that was matched
    placeholder: str   # what it was replaced with, e.g. "<EMAIL_1>"


@dataclass
class AnonymizeResult:
    original: str
    anonymized: str
    entities: list[PIIEntity]

    @property
    def has_pii(self) -> bool:
        return len(self.entities) > 0


def scan(text: str) -> AnonymizeResult:
    """
    Detect PII in *text* and return an AnonymizeResult.

    If Presidio + spaCy are available, uses full NLP-based detection.
    Otherwise falls back to regex patterns for structured PII
    (emails, phones, credit cards, IPs, SSNs, IBANs).
    """
    if _PRESIDIO_AVAILABLE:
        return _presidio_scan(text)
    return _regex_scan(text)


def _presidio_scan(text: str) -> AnonymizeResult:
    results = _analyzer.analyze(text=text, language="en")
    if not results:
        return AnonymizeResult(original=text, anonymized=text, entities=[])

    # Build placeholder map: entity_type → counter
    counters: dict[str, int] = {}
    entities: list[PIIEntity] = []

    # Sort by position so we replace left-to-right
    for r in sorted(results, key=lambda x: x.start):
        original_token = text[r.start:r.end]
        counters[r.entity_type] = counters.get(r.entity_type, 0) + 1
        placeholder = f"<{r.entity_type}_{counters[r.entity_type]}>"
        entities.append(PIIEntity(
            entity_type=r.entity_type,
            original=original_token,
            placeholder=placeholder,
        ))

    anonymized = text
    # Replace longest matches first (avoid offset drift)
    for entity in sorted(entities, key=lambda e: -len(e.original)):
        anonymized = anonymized.replace(entity.original, entity.placeholder, 1)

    return AnonymizeResult(original=text, anonymized=anonymized, entities=entities)


def _regex_scan(text: str) -> AnonymizeResult:
    entities: list[PIIEntity] = []
    anonymized = text
    counters: dict[str, int] = {}

    for entity_type, pattern, prefix in _PATTERNS:
        for match in re.finditer(pattern, anonymized):
            counters[prefix] = counters.get(prefix, 0) + 1
            placeholder = f"<{prefix}_{counters[prefix]}>"
            original = match.group()
            entities.append(PIIEntity(
                entity_type=entity_type,
                original=original,
                placeholder=placeholder,
            ))
            anonymized = anonymized.replace(original, placeholder, 1)

    return AnonymizeResult(original=text, anonymized=anonymized, entities=entities)


def presidio_available() -> bool:
    return _PRESIDIO_AVAILABLE
