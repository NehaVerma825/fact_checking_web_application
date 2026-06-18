import os
import json
import fitz
import pandas as pd
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from duckduckgo_search import DDGS

# ==================================================
# CONFIG
# ==================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# ==================================================
# STREAMLIT PAGE
# ==================================================

st.set_page_config(
    page_title="Fact Check Agent",
    layout="wide"
)

st.title("🔍 Fact Check Agent")
st.markdown(
    """
Upload a PDF and automatically verify factual claims
against live web search results.
"""
)

# ==================================================
# GEMINI HELPER
# ==================================================

def ask_llm(prompt):

    try:

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:

        return str(e)

# ==================================================
# PDF EXTRACTION
# ==================================================

def extract_pdf_text(uploaded_file):

    pdf_bytes = uploaded_file.read()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    text = ""

    for page in doc:
        text += page.get_text()

    return text

# ==================================================
# CLAIM EXTRACTION
# ==================================================

def extract_claims(text):

    prompt = f"""
You are a claim extraction engine.

Extract ONLY factual claims.

Focus on:

- percentages
- statistics
- dates
- financial figures
- revenue
- market share
- user counts
- technical figures

Return ONLY valid JSON.

Example:

[
 {{
   "claim":"OpenAI was founded in 2015",
   "search_query":"OpenAI founded year"
 }},
 {{
   "claim":"India GDP growth was 12%",
   "search_query":"India GDP growth 2024 IMF"
 }}
]

Document:

{text[:15000]}
"""

    try:

        response = ask_llm(prompt)

        response = (
            response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        claims = json.loads(response)

        return claims

    except Exception as e:

        st.error(
            f"Claim Extraction Error: {e}"
        )

        return []

# ==================================================
# DUCKDUCKGO SEARCH
# ==================================================

def search_web(query):

    evidence = ""
    urls = []

    try:

        with DDGS() as ddgs:

            results = ddgs.text(
                query,
                max_results=5
            )

            for r in results:

                title = r.get(
                    "title",
                    ""
                )

                body = r.get(
                    "body",
                    ""
                )

                href = r.get(
                    "href",
                    ""
                )

                urls.append(href)

                evidence += f"""

Title:
{title}

Snippet:
{body}

URL:
{href}

--------------------------------
"""

    except Exception as e:

        evidence = f"Search Error: {e}"

    return evidence, urls

# ==================================================
# FACT CHECKING
# ==================================================

def verify_claim(
    claim,
    evidence
):

    prompt = f"""
You are an expert fact checker.

Claim:

{claim}

Evidence:

{evidence}

Classify as:

1. Verified
2. Inaccurate
3. False

Definitions:

Verified:
Evidence clearly supports claim.

Inaccurate:
Partially true, outdated,
or wrong number/date.

False:
Evidence contradicts claim
or no reliable evidence exists.

Return ONLY JSON.

{{
  "status":"",
  "correct_fact":"",
  "reason":"",
  "confidence":0
}}

Confidence must be
between 0 and 100.
"""

    try:

        response = ask_llm(prompt)

        response = (
            response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        result = json.loads(response)

        return result

    except Exception as e:

        return {
            "status": "Unknown",
            "correct_fact": "",
            "reason": str(e),
            "confidence": 0
        }

# ==================================================
# MAIN APP
# ==================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner(
        "Extracting PDF..."
    ):

        text = extract_pdf_text(
            uploaded_file
        )

    st.success(
        "PDF loaded successfully"
    )

    with st.expander(
        "Preview Extracted Text"
    ):

        st.write(
            text[:5000]
        )

    if st.button(
        "🚀 Start Fact Checking"
    ):

        with st.spinner(
            "Extracting claims..."
        ):

            claims = extract_claims(
                text
            )

        if len(claims) == 0:

            st.error(
                "No claims found."
            )

            st.stop()

        st.success(
            f"{len(claims)} claims detected."
        )

        results = []

        progress = st.progress(0)

        for index, item in enumerate(
            claims
        ):

            claim = item.get(
                "claim",
                ""
            )

            search_query = item.get(
                "search_query",
                claim
            )

            evidence, urls = search_web(
                search_query
            )

            verification = verify_claim(
                claim,
                evidence
            )

            results.append(
                {
                    "Claim": claim,
                    "Status": verification.get(
                        "status"
                    ),
                    "Confidence": verification.get(
                        "confidence"
                    ),
                    "Correct Fact": verification.get(
                        "correct_fact"
                    ),
                    "Reason": verification.get(
                        "reason"
                    ),
                    "Sources": "\n".join(
                        urls
                    )
                }
            )

            progress.progress(
                (index + 1)
                / len(claims)
            )

        df = pd.DataFrame(
            results
        )

        st.subheader(
            "📊 Fact Check Results"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        verified = len(
            df[
                df["Status"]
                == "Verified"
            ]
        )

        inaccurate = len(
            df[
                df["Status"]
                == "Inaccurate"
            ]
        )

        false_count = len(
            df[
                df["Status"]
                == "False"
            ]
        )

        st.subheader(
            "📈 Summary"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Verified",
            verified
        )

        c2.metric(
            "Inaccurate",
            inaccurate
        )

        c3.metric(
            "False",
            false_count
        )

        csv = df.to_csv(
            index=False
        )

        st.download_button(
            "⬇ Download Report",
            csv,
            "fact_check_report.csv",
            "text/csv"
        )

        st.success(
            "Fact checking completed."
        )

