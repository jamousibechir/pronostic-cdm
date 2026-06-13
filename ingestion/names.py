"""
Normalisation centrale des noms d'équipes.

Espace canonique = noms martj42 (corpus d'entraînement de 49k matchs).
Toutes les autres sources (football-data.org pour les fixtures, eloratings.net
pour le prior Elo) sont normalisées VERS cet espace, sinon le modèle s'entraîne
sur « United States » mais prédit « USA » et retombe sur les ratings par défaut.

Usage :
    from ingestion.names import canonical
    df["home_team"] = df["home_team"].map(canonical)
"""

# Alias source -> nom canonique martj42
# (seules les divergences réelles sont listées ; tout le reste passe tel quel)
_ALIASES: dict[str, str] = {
    # ── football-data.org (fixtures WC 2026) ──────────────────────────────────
    "Bosnia-Herzegovina":   "Bosnia and Herzegovina",
    "Cape Verde Islands":   "Cape Verde",
    "Congo DR":             "DR Congo",
    "Czechia":              "Czech Republic",

    # ── Continuités historiques (lignée de fédération) ────────────────────────
    "West Germany":         "Germany",   # FIFA attribue les titres RFA à l'Allemagne

    # ── eloratings.net / FIFA -> martj42 ──────────────────────────────────────
    "USA":                  "United States",
    "Korea Republic":       "South Korea",
    "Korea DPR":            "North Korea",
    "IR Iran":              "Iran",
    "Côte d'Ivoire":        "Ivory Coast",
    "Cote d'Ivoire":        "Ivory Coast",
    "Cabo Verde":           "Cape Verde",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Türkiye":              "Turkey",
    "Turkiye":              "Turkey",
    "China PR":             "China",
    "Chinese Taipei":       "Taiwan",
    "Kyrgyz Republic":      "Kyrgyzstan",
    "St. Kitts & Nevis":    "Saint Kitts and Nevis",
    "St. Vincent & the Grenadines": "Saint Vincent and the Grenadines",
    "St. Lucia":            "Saint Lucia",
    "Trinidad & Tobago":    "Trinidad and Tobago",
    "Antigua & Barbuda":    "Antigua and Barbuda",
    "São Tomé & Príncipe":  "São Tomé and Príncipe",
    "Curaçao":              "Curacao",
    "St. Martin":           "Saint Martin",
    "Brunei Darussalam":    "Brunei",
    "Cabo Verde Islands":   "Cape Verde",
    "Republic of Ireland":  "Republic of Ireland",
    "North Macedonia":      "North Macedonia",
    "Congo":                "Congo",
    "DR Congo":             "DR Congo",
}


def canonical(name) -> str:
    """Retourne le nom canonique martj42 pour une équipe (idempotent)."""
    if name is None:
        return ""
    s = str(name).strip()
    return _ALIASES.get(s, s)
