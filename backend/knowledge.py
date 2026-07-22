"""Rule-based treatment recommendation module (paper Step 5).

Deliberately free of any TensorFlow import: the paper describes this module as
operating independently of the image classifier, so /recommend must keep working
even when the model file is missing.

Matching semantics are carried over from the notebook -- both sides lowercased,
longest phrase first, whole-word regex -- with one deliberate fix. The notebook
indexed `term -> disease` in a plain dict, so a term owned by several diseases
kept only the last one inserted. 'stunting' belongs to BYDV, WSBMV and WSMV, but
only WSMV survived, silently hiding two candidates. Here the index maps
`term -> [diseases]` so every owner is returned.
"""

import json
import re
from pathlib import Path

Database = dict[str, dict]
TermIndex = dict[str, list[str]]


def load_db(path: Path) -> Database:
    """Load and validate the knowledge base JSON."""
    with open(path, encoding="utf-8") as fh:
        db = json.load(fh)

    if not isinstance(db, dict) or not db:
        raise ValueError(f"{path}: expected a non-empty object")
    for name, entry in db.items():
        missing = {"common_names", "symptoms", "recommendation"} - set(entry)
        if missing:
            raise ValueError(f"{path}: entry {name!r} is missing {sorted(missing)}")
    return db


def build_term_index(db: Database) -> TermIndex:
    """Map every searchable term to the list of diseases that claim it.

    Ordered longest-term-first so multi-word phrases are reported before the
    single words they contain ('yellow stripes' before 'yellow rust').
    """
    index: TermIndex = {}
    for disease, info in db.items():
        for term in info["common_names"] + info["symptoms"]:
            owners = index.setdefault(term.lower(), [])
            if disease not in owners:
                owners.append(disease)
    return dict(sorted(index.items(), key=lambda kv: -len(kv[0])))


# Terms this short are the paper's acronyms (YR, BR, PM, SLB, TS, HB, LS, KB, RR,
# LB). Matched case-insensitively they fire on ordinary prose -- "we applied 50 lb
# of seed" reads as Leaf_Blight, "the pm reading" as Powdery_Mildew. So a term
# below this length must appear in UPPERCASE in the original text to count. "YR"
# still matches; "yr over yr growth" no longer does.
ACRONYM_MAX_LEN = 3


def extract_terms(text: str, index: TermIndex) -> list[tuple[str, list[str]]]:
    """Find every known term appearing in `text` as a whole word.

    Returns (term, [diseases]) pairs in index order (longest term first). Every
    term is tested -- this is not longest-match-wins, so a query can legitimately
    match both 'yellow rust' and 'yellow stripes'.
    """
    lowered = text.lower()
    matches: list[tuple[str, list[str]]] = []
    for term, diseases in index.items():
        # Short acronyms are matched against the original text, case-sensitively.
        haystack, needle = (
            (text, term.upper()) if len(term) <= ACRONYM_MAX_LEN else (lowered, term)
        )
        # \b so 'rust' does not match inside 'trust'
        if re.search(rf"\b{re.escape(needle)}\b", haystack):
            matches.append((term, diseases))
    return matches


def recommend_from_text(text: str, db: Database, index: TermIndex) -> dict:
    """Full pipeline: extraction -> alignment -> treatment recommendations."""
    matches = extract_terms(text, index)

    # Preserve first-appearance order while collecting which terms hit each disease.
    terms_by_disease: dict[str, list[str]] = {}
    for term, diseases in matches:
        for disease in diseases:
            terms_by_disease.setdefault(disease, []).append(term)

    if not terms_by_disease:
        return {
            "diseases": [],
            "recommendations": [],
            "ambiguous": False,
            "message": "No known disease or symptom recognized. "
            "Try describing the symptom differently, or upload an image.",
        }

    # A single term owned by several diseases means we are returning candidates,
    # not a diagnosis -- 'stunting' alone cannot distinguish BYDV from WSBMV or
    # WSMV. The UI uses this to say so plainly.
    ambiguous = any(len(diseases) > 1 for _, diseases in matches)

    return {
        "diseases": list(terms_by_disease),
        "recommendations": [
            {
                "disease": disease,
                "matched_terms": terms,
                "symptoms": db[disease]["symptoms"],
                "recommendation": db[disease]["recommendation"],
            }
            for disease, terms in terms_by_disease.items()
        ],
        "ambiguous": ambiguous,
        "message": "ok",
    }
