"""Pure label canonicalisation toward the domain taxonomy.

Translated from the PR #10 sandbox (`LABEL_ALIASES` + `canonicalize`). No I/O,
no third-party imports, so it is safe to import without any optional extra.
"""

import re

_LABEL_ALIASES: dict[str, str] = {
    "first_name": "person",
    "last_name": "person",
    "date": "date_time",
    "date_of_birth": "date_time",
    "birthday": "date_time",
    "email": "email_address",
    "phone": "phone_number",
    "address": "street_address",
    "location": "street_address",
    "ip": "ip_address",
    "ssn": "us_ssn",
    "bank_routing_number": "iban_code",
    "account_number": "iban_code",
    "credit_card_number": "credit_card",
    "mac_address": "ip_address",
}

_IRRELEVANT = {"o", "outside", "none", "other", "unknown"}
_SCHEME_PREFIX = re.compile(r"^(B|I|E|S|U|L)[-_]", re.IGNORECASE)
_GENERIC_LABEL = re.compile(r"^label_\d+$")


def canonicalize(label: str) -> str:
    """Normalise an entity label: strip BIO/BILOU prefix, lowercase, apply aliases."""
    if not label:
        return "unknown"
    normalized = _SCHEME_PREFIX.sub("", str(label).strip())
    normalized = normalized.lower().replace(" ", "_").replace("-", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return _LABEL_ALIASES.get(normalized, normalized)


def supported_classes_from_id2label(id2label: dict[int, str] | None) -> frozenset[str]:
    """Project a model's ``id2label`` map onto canonical domain labels, dropping generic tags."""
    if not isinstance(id2label, dict):
        return frozenset()
    classes = {
        label
        for label in (canonicalize(raw) for raw in id2label.values())
        if label not in _IRRELEVANT and not _GENERIC_LABEL.match(label)
    }
    return frozenset(classes)
