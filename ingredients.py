"""Shared key-ingredient evidence: "shares niacinamide, hyaluronic acid".

Why a curated active list instead of a raw INCI intersection: full
ingredient lists overlap heavily on base and vehicle ingredients (water,
glycerin, tocopherol, phenoxyethanol, titanium dioxide) that appear in a
large fraction of the catalog and tell a reader nothing about why two
products are related. Reporting those as "shared ingredients" would be
technically true and completely uninformative. The curated list below is
restricted to actives a beauty shopper would recognize and actually choose
a product for, and each was checked against this catalog to confirm it
appears at an informative rate (roughly 0.1%-20% of products; anything
near-universal was deliberately excluded).

Matching is by regex over the raw ingredient text rather than by exact
token equality because the same active appears under many INCI spellings
and salt/derivative forms -- "hyaluronic acid" is variously Sodium
Hyaluronate, Hydrolyzed Hyaluronic Acid, Sodium Acetylated Hyaluronate.
Word boundaries matter: a naive substring test for "urea" also matches
polyurea, and "pha" matches alpha/sulphate, so short tokens are anchored.
"""

from __future__ import annotations

import re

import pandas as pd

# label -> regex matched (case-insensitively) against the raw ingredient text
KEY_ACTIVES: dict[str, str] = {
    "niacinamide": r"niacinamide",
    "hyaluronic acid": r"hyaluron",
    "salicylic acid": r"salicylic acid",
    "glycolic acid": r"glycolic acid",
    "lactic acid": r"lactic acid",
    "mandelic acid": r"mandelic",
    "azelaic acid": r"azelaic",
    "tranexamic acid": r"tranexamic",
    "kojic acid": r"kojic",
    "retinol": r"retinol",
    "bakuchiol": r"bakuchiol",
    "vitamin C": r"ascorbic|ascorbyl",
    "vitamin E": r"\btocopheryl acetate\b",
    "ceramides": r"ceramide",
    "peptides": r"peptide",
    "collagen": r"collagen",
    "squalane": r"squalane",
    "shea butter": r"shea|butyrospermum",
    "jojoba": r"jojoba",
    "argan oil": r"argan",
    "rosehip": r"rosa canina|rosehip",
    "aloe": r"aloe",
    "green tea": r"camellia sinensis|green tea",
    "centella": r"centella|madecassoside|cica\b",
    "licorice root": r"licorice|glycyrrhiza",
    "turmeric": r"turmeric|curcuma",
    "witch hazel": r"hamamelis|witch hazel",
    "tea tree": r"tea tree|melaleuca",
    "oat extract": r"avena sativa",
    "panthenol": r"panthenol",
    "allantoin": r"allantoin",
    "adenosine": r"adenosine",
    "caffeine": r"caffeine",
    "urea": r"\burea\b",
    "arbutin": r"arbutin",
    "resveratrol": r"resveratrol",
    "gluconolactone": r"gluconolactone",
    "snail mucin": r"snail",
    "charcoal": r"charcoal",
    "zinc oxide": r"zinc oxide",
}

_COMPILED = {label: re.compile(rx, re.IGNORECASE) for label, rx in KEY_ACTIVES.items()}


def actives_in(ingredient_text: object) -> set[str]:
    """The curated actives present in one product's ingredient text. Returns
    an empty set for missing/non-string ingredients (945 of the 8,494
    catalog products have no ingredient data at all) rather than raising —
    absence of ingredient data is a normal, expected state here, and the
    caller distinguishes "no shared actives" from "no data" by checking
    whether the product has ingredients at all."""
    if not isinstance(ingredient_text, str) or not ingredient_text.strip():
        return set()
    return {label for label, rx in _COMPILED.items() if rx.search(ingredient_text)}


def build_actives_lookup(products: pd.DataFrame) -> dict[str, set[str]]:
    """product_id -> set of curated actives, for the whole catalog."""
    return {
        row.product_id: actives_in(row.ingredients)
        for row in products[["product_id", "ingredients"]].itertuples(index=False)
    }


def shared_actives(
    product_id_a: str, product_id_b: str, lookup: dict[str, set[str]]
) -> list[str]:
    """Curated actives present in both products, sorted for stable display."""
    return sorted(lookup.get(product_id_a, set()) & lookup.get(product_id_b, set()))


def shared_actives_text(shared: list[str], max_shown: int = 3) -> str | None:
    """Human-readable evidence string, or None when there's nothing to say.
    Returns None rather than an empty/filler string so callers must decide
    explicitly whether to render anything — an ingredient line that says
    "shares nothing" is worse than no line."""
    if not shared:
        return None
    shown = shared[:max_shown]
    extra = len(shared) - len(shown)
    text = "Shares " + ", ".join(shown)
    if extra > 0:
        text += f" +{extra} more"
    return text
