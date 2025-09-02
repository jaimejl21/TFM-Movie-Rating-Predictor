import os
import json
import time
import re
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Config & Secrets
# -----------------------------
SERVING_URL = st.secrets.get("SERVING_URL", os.getenv("SERVING_URL", ""))
DATABRICKS_TOKEN = st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN", ""))

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

st.set_page_config(page_title="Movie Rating Predictor (TFM)", page_icon="🎬", layout="centered")

st.title("🎬 TFM — Movie Rating Predictor")
st.write(
    """
    This tool helps users find the best main actor, main director and main writer for a movie,
    given selected genres and a brief plot. Fill in the details below.
    """
)

# -----------------------------
# Helpers (sanitation & validation)
# -----------------------------
def _split_csv(s: str):
    """Split by comma, strip whitespace, drop empties and deduplicate preserving order."""
    if not s:
        return []
    seen = set()
    out = []
    for x in [t.strip() for t in s.split(",") if t.strip()]:
        if x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out

def _has_letter(s: str) -> bool:
    """At least one alphabetic character (handles unicode letters by any .isalpha char)."""
    return any(ch.isalpha() for ch in s)

def _too_many_symbols(s: str, threshold: float = 0.5) -> bool:
    """Heuristic: if more than threshold of chars are non-alnum and not space/.-'(), consider gibberish."""
    allowed_punct = set(" .,'()-")
    if not s:
        return True
    nonsafe = sum(1 for ch in s if not (ch.isalnum() or ch in allowed_punct))
    return (nonsafe / len(s)) > threshold

def validate_name_list(items, label, min_len=1, max_items=30, item_min_chars=2, item_max_chars=80):
    """
    Validate a list of names (actors/directors/writers).
    - Non-empty list
    - Each item has letters, reasonable length, not too many symbols
    - Limit total items to avoid huge payloads
    Returns: (clean_items, errors_list)
    """
    errors = []
    clean = []

    if len(items) < min_len:
        errors.append(f"• {label}: provide at least {min_len} value.")

    if len(items) > max_items:
        errors.append(f"• {label}: too many values ({len(items)}). Max allowed: {max_items}.")
        items = items[:max_items]

    for it in items:
        s = re.sub(r"\s+", " ", it).strip()
        if not s:
            continue
        if len(s) < item_min_chars:
            errors.append(f"• {label}: “{it}” is too short.")
            continue
        if len(s) > item_max_chars:
            errors.append(f"• {label}: “{it}” is too long.")
            continue
        if not _has_letter(s):
            errors.append(f"• {label}: “{it}” must contain letters.")
            continue
        if _too_many_symbols(s):
            errors.append(f"• {label}: “{it}” looks invalid (too many symbols).")
            continue
        clean.append(s)

    if len(clean) < min_len:
        errors.append(f"• {label}: no valid values after cleaning.")
    return clean, errors

# -----------------------------
# Inputs
# -----------------------------
COMMON_GENRES = [
    "Action","Adventure","Comedy","Drama","Romance","Sci-Fi","Thriller","Horror","Fantasy","Mystery",
    "Crime","Animation","Family","Biography","History","Music","War","Western","Sport","Documentary"
]
genres = st.multiselect("🎭 Genres", COMMON_GENRES, default=["Comedy", "Romance"])

# NOTE: max_chars impone el límite duro de 100 chars
plot = st.text_area(
    "📝 Plot (100 char max)",
    "A light-hearted office romance.",
    height=100,
    max_chars=100,
    help="Short logline or brief synopsis. 100 characters maximum.",
)
st.caption(f"Characters: {len(plot)}/100")

actors_csv    = st.text_input("🧑‍🎤 Actors (comma separated)",    "Jack Lemmon, Will Smith, Walter Matthau")
directors_csv = st.text_input("🎬 Directors (comma separated)",     "Billy Wilder, Roger Michell")
writers_csv   = st.text_input("✍️ Writers (comma separated)",       "Nora Ephron, Richard Curtis")
top_k         = st.number_input("🔢 Top K combinations", min_value=1, max_value=50, value=12, step=1)

actors    = _split_csv(actors_csv)
directors = _split_csv(directors_csv)
writers   = _split_csv(writers_csv)

# -----------------------------
# Robust request with retry/backoff
# -----------------------------
def call_with_retry(url, headers, payload, max_retries=4, connect_timeout=15, read_timeout=300):
    backoff = 2
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=(connect_timeout, read_timeout))
            if resp.status_code >= 500:
                st.info(f"[Try {attempt}] HTTP {resp.status_code}. Retrying in {backoff}s…")
                time.sleep(backoff)
                backoff *= 2
                continue
            return resp
        except (requests.ReadTimeout, requests.ConnectionError) as e:
            last_exc = e
            st.info(f"[Try {attempt}] {type(e).__name__}: {e}. Retrying in {backoff}s…")
            time.sleep(backoff)
            backoff *= 2
    if last_exc:
        raise last_exc
    raise RuntimeError("Request failed without exception")

# -----------------------------
# Predict
# -----------------------------
if st.button("🚀 Predict rating"):
    errors = []

    # --- Required fields ---
    if not SERVING_URL or not DATABRICKS_TOKEN:
        errors.append("• Secrets missing: configure SERVING_URL and DATABRICKS_TOKEN.")

    if not genres:
        errors.append("• Genres: select at least one genre.")

    plot_clean = (plot or "").strip()
    if not plot_clean:
        errors.append("• Plot: cannot be empty.")
    if len(plot_clean) > 100:
        errors.append("• Plot: must be ≤ 100 characters.")

    # --- Validate name lists ---
    actors_clean, err_a = validate_name_list(actors, "Actors")
    directors_clean, err_d = validate_name_list(directors, "Directors")
    writers_clean, err_w = validate_name_list(writers, "Writers")
    errors.extend(err_a + err_d + err_w)

    # --- Optional: cap top_k to theoretical maximum combinations ---
    max_combos = max(1, len(actors_clean) * len(directors_clean) * len(writers_clean))
    if top_k > max_combos:
        st.info(f"Top K reduced from {int(top_k)} to {max_combos} (max possible with current lists).")
        top_k = max_combos

    # --- If any error, show and stop ---
    if errors:
        st.error("Please fix the following issues:\n" + "\n".join(errors))
        st.stop()

    # --- Build payload ---
    row = {
        "genres": genres,
        "plot": plot_clean,
        "actors": actors_clean,
        "directors": directors_clean,
        "writers": writers_clean,
        "top_k": int(top_k),
    }
    payload = {"dataframe_records": [row]}

    # --- Call endpoint ---
    with st.spinner("Calling Databricks Model Serving… (first try may take longer due to cold start)"):
        try:
            resp = call_with_retry(SERVING_URL, HEADERS, payload)
            if resp.status_code != 200:
                st.error(f"HTTP {resp.status_code}")
                st.code(resp.text[:1200])
            else:
                js = resp.json()
                pred_list = []
                if "predictions" in js and js["predictions"]:
                    pred_list = js["predictions"][0].get("predictions", [])

                df = pd.DataFrame(pred_list)
                if df.empty:
                    st.warning("⚠️ No predictions received.")
                else:
                    df = df.rename(columns={
                        "combination": "Combination",
                        "rating_estimado": "Estimated Rating",
                        "rating": "Estimated Rating",
                    })
                    if "Estimated Rating" in df.columns:
                        df["Estimated Rating"] = pd.to_numeric(df["Estimated Rating"], errors="coerce").round(2)
                    df.insert(0, "#", range(1, len(df) + 1))
                    st.subheader("📊 Results")
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download CSV", csv, "predictions.csv", "text/csv")

                    with st.expander("🔎 View raw JSON"):
                        st.code(json.dumps(js, indent=2)[:10000])

        except Exception as e:
            st.exception(e)
