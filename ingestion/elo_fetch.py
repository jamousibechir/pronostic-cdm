"""
Récupère les ratings Elo internationaux depuis eloratings.net.

Source : https://www.eloratings.net/World.tsv (fichier TSV public)
Format TSV réel :
  col[0]=current_rank, col[1]=prev_rank, col[2]=country_code (2 lettres),
  col[3]=current_elo, [col[4..30] = données historiques non utilisées]

Les codes eloratings.net NE sont PAS des codes ISO standard :
  EN=England, SQ=Scotland, WA=Wales, KO=Kosovo, NM=North Macedonia, etc.

Sortie : data/elo_ratings.csv
  Colonnes : team, elo, rank
"""
import requests
import pandas as pd
from io import StringIO

from config import DATA_DIR, ELO_URL

HEADERS = {"User-Agent": "pronostic-cdm/1.0 (research project)"}
OUTPUT  = DATA_DIR / "elo_ratings.csv"

# Mapping complet : code eloratings.net → nom FIFA officiel
# Codes non-ISO signalés : EN, SQ, WA, KO, NM, NS, EI, ZN, KD, etc.
ELO_CODE_TO_NAME: dict[str, str] = {
    # ── Top 20 ──────────────────────────────────────────────────────────────
    "ES": "Spain",
    "AR": "Argentina",
    "FR": "France",
    "EN": "England",
    "BR": "Brazil",
    "PT": "Portugal",
    "CO": "Colombia",
    "NL": "Netherlands",
    "EC": "Ecuador",
    "DE": "Germany",
    "NO": "Norway",
    "HR": "Croatia",
    "TR": "Türkiye",
    "JP": "Japan",
    "BE": "Belgium",
    "UY": "Uruguay",
    "CH": "Switzerland",
    "MX": "Mexico",
    "DK": "Denmark",
    "IT": "Italy",
    # ── 21-50 ────────────────────────────────────────────────────────────────
    "SN": "Senegal",
    "PY": "Paraguay",
    "AT": "Austria",
    "MA": "Morocco",
    "CA": "Canada",
    "SQ": "Scotland",         # non-ISO (eloratings.net)
    "UA": "Ukraine",
    "AU": "Australia",
    "IR": "IR Iran",
    "RU": "Russia",
    "NG": "Nigeria",
    "DZ": "Algeria",
    "KR": "Korea Republic",
    "GR": "Greece",
    "CZ": "Czech Republic",
    "RS": "Serbia",
    "VE": "Venezuela",
    "PA": "Panama",
    "US": "USA",
    "CL": "Chile",
    "KO": "Kosovo",           # non-ISO (eloratings.net)
    "UZ": "Uzbekistan",
    "SE": "Sweden",
    "HU": "Hungary",
    "PL": "Poland",
    "PE": "Peru",
    "IE": "Republic of Ireland",
    "EG": "Egypt",
    "CI": "Côte d'Ivoire",
    "WA": "Wales",            # non-ISO (eloratings.net)
    # ── 51-100 ───────────────────────────────────────────────────────────────
    "SI": "Slovenia",
    "JO": "Jordan",
    "SK": "Slovakia",
    "GE": "Georgia",
    "CD": "DR Congo",
    "IL": "Israel",
    "RO": "Romania",
    "BO": "Bolivia",
    "TN": "Tunisia",
    "AL": "Albania",
    "CM": "Cameroon",
    "CR": "Costa Rica",
    "IQ": "Iraq",
    "EI": "Republic of Ireland",  # code historique Éire (doublon possible)
    "BA": "Bosnia & Herzegovina",
    "NM": "North Macedonia",  # non-ISO (eloratings.net ; ISO = MK)
    "ML": "Mali",
    "CV": "Cabo Verde",
    "SA": "Saudi Arabia",
    "HN": "Honduras",
    "IS": "Iceland",
    "NZ": "New Zealand",
    "HT": "Haiti",
    "AO": "Angola",
    "AE": "United Arab Emirates",
    "FI": "Finland",
    "BF": "Burkina Faso",
    "JM": "Jamaica",
    "BY": "Belarus",
    "ZA": "South Africa",
    "GH": "Ghana",
    "GT": "Guatemala",
    "OM": "Oman",
    "SY": "Syria",
    "PS": "Palestine",
    "GN": "Guinea",
    "ME": "Montenegro",
    "BG": "Bulgaria",
    "LU": "Luxembourg",
    "NS": "Northern Ireland",  # non-ISO (eloratings.net)
    "CW": "Curaçao",
    "SR": "Suriname",
    "KZ": "Kazakhstan",
    "CN": "China PR",
    "KD": "Korea DPR",
    "QA": "Qatar",
    "LY": "Libya",
    "GM": "Gambia",
    "BH": "Bahrain",
    "BJ": "Benin",
    # ── 101-150 ──────────────────────────────────────────────────────────────
    "GA": "Gabon",
    "UG": "Uganda",
    "TT": "Trinidad & Tobago",
    "FO": "Faroe Islands",
    "NE": "Niger",
    "MG": "Madagascar",
    "GQ": "Equatorial Guinea",
    "TG": "Togo",
    "TH": "Thailand",
    "KP": "Korea DPR",
    "KM": "Comoros",
    "AM": "Armenia",
    "ZW": "Zimbabwe",
    "ID": "Indonesia",
    "ZM": "Zambia",
    "KE": "Kenya",
    "EE": "Estonia",
    "VN": "Vietnam",
    "SD": "Sudan",
    "RE": "Réunion",
    "SV": "El Salvador",
    "MZ": "Mozambique",
    "SL": "Sierra Leone",
    "GP": "Guadeloupe",
    "RW": "Rwanda",
    "NI": "Nicaragua",
    "KW": "Kuwait",
    "MR": "Mauritania",
    "AZ": "Azerbaijan",
    "ZN": "Zanzibar",
    "CY": "Cyprus",
    "TZ": "Tanzania",
    "MQ": "Martinique",
    "LR": "Liberia",
    "KG": "Kyrgyzstan",
    "MY": "Malaysia",
    "GY": "Guyana",
    "LB": "Lebanon",
    "LV": "Latvia",
    "ET": "Ethiopia",
    "NC": "New Caledonia",
    "TJ": "Tajikistan",
    "BI": "Burundi",
    "DO": "Dominican Republic",
    "LT": "Lithuania",
    "MD": "Moldova",
    "BW": "Botswana",
    "MT": "Malta",
    "GW": "Guinea-Bissau",
    "CU": "Cuba",
    "MW": "Malawi",
    "CF": "Central African Republic",
    # ── 151-200 ──────────────────────────────────────────────────────────────
    "GF": "French Guiana",
    "YT": "Mayotte",
    "TM": "Turkmenistan",
    "CG": "Congo",
    "ER": "Eritrea",
    "LS": "Lesotho",
    "YE": "Yemen",
    "PH": "Philippines",
    "TI": "Chinese Taipei",
    "SW": "Eswatini",
    "VC": "St. Vincent & the Grenadines",
    "PG": "Papua New Guinea",
    "PR": "Puerto Rico",
    "SG": "Singapore",
    "IN": "India",
    "VU": "Vanuatu",
    "BM": "Bermuda",
    "SS": "South Sudan",
    "FJ": "Fiji",
    "HK": "Hong Kong",
    "GD": "Grenada",
    "AD": "Andorra",
    "MU": "Mauritius",
    "TD": "Chad",
    "BZ": "Belize",
    "SB": "Solomon Islands",
    "MF": "Saint Martin",
    "ST": "São Tomé & Príncipe",
    "KN": "St. Kitts & Nevis",
    "GI": "Gibraltar",
    "JS": "Jersey",
    "LC": "St. Lucia",
    "EH": "Western Sahara",
    "MM": "Myanmar",
    "SO": "Somalia",
    "AW": "Aruba",
    "SX": "Sint Maarten",
    "MS": "Montserrat",
    "AF": "Afghanistan",
    "GL": "Greenland",
    "BD": "Bangladesh",
    "DJ": "Djibouti",
    "DM": "Dominica",
    "PK": "Pakistan",
    "MC": "Monaco",
    "BB": "Barbados",
    "AG": "Antigua & Barbuda",
    "LI": "Liechtenstein",
    # ── 201-244 ──────────────────────────────────────────────────────────────
    "NP": "Nepal",
    "KH": "Cambodia",
    "SC": "Seychelles",
    "LK": "Sri Lanka",
    "SM": "San Marino",
    "TW": "Chinese Taipei",
    "BQ": "Bonaire",
    "MV": "Maldives",
    "KY": "Cayman Islands",
    "HG": "Northern Mariana Islands",
    "TV": "Tuvalu",
    "VG": "British Virgin Islands",
    "EU": "Timor-Leste",
    "LA": "Laos",
    "TL": "Timor-Leste",
    "WS": "Samoa",
    "MN": "Mongolia",
    "BL": "Saint Barthélemy",
    "GU": "Guam",
    "WF": "Wallis & Futuna",
    "VA": "Vatican City",
    "AB": "Abkhazia",
    "BS": "Bahamas",
    "PM": "Saint Pierre & Miquelon",
    "TC": "Turks & Caicos",
    "AI": "Anguilla",
    "TE": "Timor-Leste",
    "VI": "US Virgin Islands",
    "BT": "Bhutan",
    "CK": "Cook Islands",
    "MO": "Macau",
    "CX": "Christmas Island",
    "BN": "Brunei",
    "FK": "Falkland Islands",
    "FM": "Micronesia",
    "MH": "Marshall Islands",
    "KI": "Kiribati",
    "TO": "Tonga",
    "NU": "Niue",
    "MP": "Northern Mariana Islands",
    "CC": "Cocos Islands",
    "PW": "Palau",
    "AS": "American Samoa",
}

# Normalisation finale → noms utilisés dans le reste du système
_FINAL_NAME_MAP = {
    "United States":    "USA",
    "IR Iran":          "IR Iran",
    "Korea Republic":   "Korea Republic",
    "Korea DPR":        "Korea DPR",
    "Côte d'Ivoire":    "Côte d'Ivoire",
    "DR Congo":         "DR Congo",
    "Republic of Ireland": "Republic of Ireland",
}


def _fetch_tsv() -> pd.DataFrame | None:
    """Télécharge et parse correctement le TSV eloratings.net."""
    try:
        r = requests.get(ELO_URL, headers=HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(
            StringIO(r.text), sep="\t", header=None, on_bad_lines="skip"
        )
        # Format réel : rank | prev_rank | code | elo | [historique...]
        df = df.iloc[:, [0, 2, 3]]
        df.columns = ["rank", "code", "elo"]
        df["elo"]  = pd.to_numeric(df["elo"],  errors="coerce")
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.dropna(subset=["code", "elo", "rank"])
        df["code"] = df["code"].astype(str).str.strip()
        # Convertit le code 2-lettres en nom complet
        df["team"] = df["code"].map(ELO_CODE_TO_NAME)
        # Pour les codes inconnus, utilise le code lui-même
        mask_unknown = df["team"].isna()
        df.loc[mask_unknown, "team"] = df.loc[mask_unknown, "code"]
        df = df[["team", "elo", "rank"]].dropna(subset=["elo", "rank"])
        df["elo"]  = df["elo"].astype(int)
        df["rank"] = df["rank"].astype(int)
        return df
    except Exception as e:
        print(f"  TSV eloratings.net : erreur ({e})")
        return None


def fetch_elo(force: bool = False) -> pd.DataFrame:
    """
    Charge les ratings Elo. Utilise le cache si disponible et force=False.

    Returns
    -------
    pd.DataFrame : colonnes [team, elo, rank]
    """
    if OUTPUT.exists() and not force:
        return pd.read_csv(OUTPUT)

    print("  Telechargement eloratings.net World.tsv...")
    df = _fetch_tsv()

    if df is None or df.empty:
        print("  Avertissement : Elo ratings indisponibles (non bloquant)")
        return pd.DataFrame(columns=["team", "elo", "rank"])

    df = df.sort_values("rank").reset_index(drop=True)
    df.to_csv(OUTPUT, index=False)
    print(f"  Elo ratings : {len(df)} equipes -> {OUTPUT}")
    return df


if __name__ == "__main__":
    df = fetch_elo(force=True)
    print(df.head(30).to_string(index=False))
