"""Outfit recommender: TF-IDF text similarity + exact attribute boost.
Light enough for Render's free 512 MB instance (no PyTorch needed)."""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("outfits.csv")
ATTRS = ["occasion", "color", "pattern", "style", "weather"]

SYNONYMS = {
    "hot": "bold", "stylish": "fashionable", "trendy": "fashionable",
    "classy": "formal", "elegant": "formal", "ethnic": "traditional",
    "strip": "striped", "stripes": "striped", "strips": "striped",
    "checks": "checked", "check": "checked", "flowers": "floral",
    "marriage": "wedding", "ceremony": "wedding", "reception": "wedding",
    "diwali": "festival", "eid": "festival", "navratri": "festival", "puja": "festival",
    "dinner": "date", "job": "interview", "work": "office", "meeting": "office",
    "college": "casual", "daily": "casual", "everyday": "casual", "brunch": "casual",
    "birthday": "party", "function": "party",
    "cold": "winter", "sunny": "summer", "rain": "rainy", "monsoon": "rainy",
    "plain": "solid", "sober": "simple",
}
NEGATORS = {"no", "not", "without", "except", "avoid", "t"}   # "t" comes from don't

VALUES = {a: set(df[a].unique()) for a in ATTRS}
vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
M = vec.fit_transform(df["description"])

# ---- colours / drawings for the website -------------------------------
HEX = {
    "beige": "#D8C3A5", "black": "#222226", "blue": "#1E5FBF", "brown": "#7B4B2A",
    "burgundy": "#6D1A36", "cream": "#EFE6D2", "gold": "#D9A93B", "green": "#2E8B57",
    "grey": "#8A8A92", "lavender": "#B7A6D9", "maroon": "#7A1F2B", "multicolor": "#D7263D",
    "mustard": "#D4A017", "navy": "#1F2A44", "olive": "#6B7B3A", "orange": "#E8742A",
    "peach": "#F4B393", "pink": "#E78FB3", "purple": "#7B3FA0", "red": "#C62828",
    "silver": "#B8BCC4", "teal": "#1F8A8A", "turquoise": "#2CC4C4", "white": "#F5F5F2",
    "yellow": "#F2C230", "charcoal": "#2E2E33", "khaki": "#CBB994", "denim": "#3C5A8A",
    "ivory": "#F3EEDC", "golden": "#D9A93B",
}
PATTERN = {"solid": "Plain", "printed": "Dotted", "striped": "Striped", "embroidered": "Floral",
           "checked": "Checks", "floral": "Floral", "embellished": "Polka"}
# (gender, occasion) -> (drawing category, variant index) from Outfit_Gallery.html
LOOK = {
    ("men", "wedding"): ("wedding", 0), ("men", "festival"): ("wedding", 2),
    ("men", "party"): ("birthday", 0), ("men", "date"): ("birthday", 1),
    ("men", "office"): ("professional", 0), ("men", "interview"): ("professional", 1),
    ("men", "casual"): ("vacation", 0),
    ("women", "wedding"): ("wedding", 0), ("women", "festival"): ("wedding", 1),
    ("women", "party"): ("kurti", 2), ("women", "date"): ("kurti", 1),
    ("women", "office"): ("professional", 2), ("women", "interview"): ("professional", 0),
    ("women", "casual"): ("kurti", 0),
}

def _lum(h):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255

def _shade(h, f):
    return "#" + "".join(f"{min(255, int(int(h[i:i + 2], 16) * f)):02x}" for i in (1, 3, 5))

def render_params(r):
    cat, variant = LOOK[(r["gender"], r["occasion"])]
    top = HEX.get(r["color"], "#1E5FBF")
    bot = next((HEX[w] for w in re.findall(r"[a-z]+", r["bottom"].lower()) if w in HEX),
               _shade(top, 0.78))
    mark = _shade(top, 0.5) if _lum(top) > 0.6 else "#FFFFFF"
    return {"cat": cat, "variant": variant, "pattern": PATTERN.get(r["pattern"], "Plain"),
            "top_hex": top, "mark_hex": mark, "bot_hex": bot,
            "label": f"{r['color']} {r['pattern']} {r['occasion']} outfit"}

# ---- query understanding ------------------------------------------------
def extract(text):
    toks = [SYNONYMS.get(w, w) for w in re.findall(r"[a-z]+", text.lower())]
    want, avoid = {}, {}
    for i, w in enumerate(toks):
        neg = any(t in NEGATORS for t in toks[max(0, i - 2):i])
        for a in ATTRS:
            if w in VALUES[a]:
                (avoid if neg else want).setdefault(a, set()).add(w)
    return " ".join(toks), want, avoid

def recommend(query, gender, weather="any", top_n=3):
    qtext, want, avoid = extract(query)
    if weather != "any":
        want.setdefault("weather", set()).add(weather)
    score = cosine_similarity(vec.transform([qtext]), M).ravel()
    for a, vals in want.items():
        score = score + 0.15 * df[a].isin(vals).values
    for a, vals in avoid.items():
        score = score - 0.5 * df[a].isin(vals).values
    if "weather" in want:                       # all-season clothes suit any weather
        score = score + 0.05 * (df["weather"] == "all-season").values
    d = df.assign(score=score)
    d = d[d["gender"] == gender].sort_values("score", ascending=False)

    best, picked, seen_tb, seen_col = 1 + 0.15 * len(want), [], set(), set()
    for need_new_colour in (True, False):       # first try to vary colours
        for _, r in d.iterrows():
            if len(picked) == top_n:
                break
            if (r["top"], r["bottom"]) in seen_tb or r["id"] in {p["id"] for p in picked}:
                continue
            if need_new_colour and r["color"] in seen_col:
                continue
            seen_tb.add((r["top"], r["bottom"])); seen_col.add(r["color"])
            hits = [f"{a}: {r[a]}" for a in want if r[a] in want[a]]
            picked.append({
                "id": int(r["id"]), "top": r["top"], "bottom": r["bottom"],
                "shoes": r["shoes"], "watch": r["watch"],
                "why": ("Matches your " + ", ".join(hits)) if hits else "Closest match to your description",
                "match": int(max(1, min(99, round(100 * r["score"] / best)))),
                **render_params(r)})
    return picked
