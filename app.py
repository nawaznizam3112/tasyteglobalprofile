import os
import re
import json
import time
import hashlib
from typing import List, Optional
from urllib.parse import urlparse

import requests
import streamlit as st

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "MasalaAtlas AI"

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)

MAX_RESEARCH_SOURCES = 12
MAX_RECIPES_TO_EXTRACT = 7
REQUEST_TIMEOUT = 18
MAX_PAGE_CHARS = 24000


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

    .stApp {
        background: #faf8f3;
    }

    .hero {
        padding: 2rem;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                rgba(255, 153, 51, .20),
                rgba(210, 40, 40, .08),
                rgba(255, 220, 120, .12)
            );
        border: 1px solid rgba(120, 70, 20, .14);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.6rem;
        margin-bottom: .2rem;
    }

    .hero p {
        font-size: 1.05rem;
        color: #665b50;
    }

    .recipe-card {
        background: white;
        padding: 1.3rem;
        border-radius: 16px;
        border: 1px solid #e7dfd2;
        margin-bottom: 1rem;
    }

    .source-card {
        background: #fff;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e7dfd2;
        margin-bottom: .7rem;
    }

    .score {
        font-size: 1.7rem;
        font-weight: 700;
    }

    .small {
        color: #766c62;
        font-size: .85rem;
    }

    .pill {
        display: inline-block;
        padding: .25rem .65rem;
        border-radius: 100px;
        background: #f0e8dc;
        margin-right: .25rem;
        margin-bottom: .25rem;
        font-size: .78rem;
    }

    .success-box {
        background: #edf8ef;
        border-left: 4px solid #2e8b57;
        padding: 1rem;
        border-radius: 8px;
    }

    .warning-box {
        background: #fff5d8;
        border-left: 4px solid #d59b00;
        padding: 1rem;
        border-radius: 8px;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA MODELS
# ============================================================

class ResearchSource(BaseModel):
    title: str = ""
    url: str = ""
    country: str = ""
    region: str = ""
    cuisine: str = ""
    dish: str = ""

    why_relevant: str = ""
    source_quality: int = Field(default=5, ge=0, le=10)
    recipe_completeness: int = Field(default=5, ge=0, le=10)
    regional_authenticity: int = Field(default=5, ge=0, le=10)
    indian_adaptation_potential: int = Field(
        default=5,
        ge=0,
        le=10,
    )


class ResearchResult(BaseModel):
    search_summary: str = ""
    sources: List[ResearchSource] = Field(
        default_factory=list
    )


class Ingredient(BaseModel):
    name: str
    quantity: str
    notes: str = ""


class Recipe(BaseModel):
    title: str = ""

    source_url: str = ""
    source_title: str = ""

    country: str = ""
    region: str = ""
    cuisine: str = ""

    description: str = ""

    ingredients: List[Ingredient] = Field(
        default_factory=list
    )

    instructions: List[str] = Field(
        default_factory=list
    )

    prep_time_minutes: int = 0
    cook_time_minutes: int = 0
    servings: int = 1

    dietary_tags: List[str] = Field(
        default_factory=list
    )

    allergens: List[str] = Field(
        default_factory=list
    )


class AdaptedRecipe(BaseModel):
    original_title: str = ""
    indian_name: str = ""

    source_url: str = ""

    original_country: str = ""
    original_region: str = ""

    target_indian_region: str = ""

    flavour_profile: str = ""

    description: str = ""

    ingredients: List[Ingredient] = Field(
        default_factory=list
    )

    instructions: List[str] = Field(
        default_factory=list
    )

    accompaniments: List[str] = Field(
        default_factory=list
    )

    substitutions: List[str] = Field(
        default_factory=list
    )

    adaptation_notes: List[str] = Field(
        default_factory=list
    )

    batch_production_notes: List[str] = Field(
        default_factory=list
    )

    prep_time_minutes: int = 0
    cook_time_minutes: int = 0
    servings: int = 1

    spice_level: int = Field(
        default=5,
        ge=0,
        le=10,
    )

    tanginess: int = Field(
        default=5,
        ge=0,
        le=10,
    )

    sweetness: int = Field(
        default=3,
        ge=0,
        le=10,
    )

    saltiness: int = Field(
        default=5,
        ge=0,
        le=10,
    )

    richness: int = Field(
        default=5,
        ge=0,
        le=10,
    )

    smokiness: int = Field(
        default=3,
        ge=0,
        le=10,
    )

    street_food_score: int = Field(
        default=5,
        ge=0,
        le=10,
    )

    commercial_score: int = Field(
        default=5,
        ge=0,
        le=10,
    )

    estimated_food_cost_inr: float = 0

    suggested_selling_price_inr: float = 0

    menu_positioning: str = ""


class GenerationResult(BaseModel):
    research_summary: str = ""

    recipes: List[AdaptedRecipe] = Field(
        default_factory=list
    )


# ============================================================
# API
# ============================================================

@st.cache_resource
def get_gemini_client():

    api_key = None

    try:
        api_key = st.secrets.get(
            "GEMINI_API_KEY"
        )
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

    if not api_key:
        return None

    return genai.Client(
        api_key=api_key
    )


def gemini_ready():

    return get_gemini_client() is not None


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(
    text: str,
    max_chars: int = MAX_PAGE_CHARS,
):

    text = re.sub(
        r"\s+",
        " ",
        text or "",
    )

    return text.strip()[:max_chars]


def normalize_url(url: str):

    if not url:
        return ""

    url = url.strip()

    if not url.startswith(
        ("http://", "https://")
    ):
        return ""

    parsed = urlparse(url)

    if not parsed.netloc:
        return ""

    return url


def domain(url: str):

    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def url_hash(url: str):

    return hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()


# ============================================================
# WEB PAGE FETCHING
# ============================================================

@st.cache_data(
    ttl=60 * 60 * 24,
    show_spinner=False,
)
def fetch_webpage(url: str):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type",
            "",
        ).lower()

        if (
            "text/html" not in content_type
            and "application/xhtml" not in content_type
        ):
            return {
                "ok": False,
                "url": url,
                "error": "Not an HTML page.",
            }

        html = response.text

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "nav",
                "footer",
                "header",
                "aside",
            ]
        ):
            tag.decompose()

        title = ""

        if soup.title:
            title = soup.title.get_text(
                " ",
                strip=True,
            )

        visible_text = clean_text(
            soup.get_text(
                " ",
                strip=True,
            )
        )

        jsonld = []

        original_soup = BeautifulSoup(
            html,
            "html.parser",
        )

        for script in original_soup.find_all(
            "script",
            type="application/ld+json",
        ):

            raw = script.string or script.get_text()

            try:

                data = json.loads(raw)

                if isinstance(data, list):
                    jsonld.extend(data)

                else:
                    jsonld.append(data)

            except Exception:
                continue

        return {
            "ok": True,
            "url": response.url,
            "title": title,
            "text": visible_text,
            "jsonld": jsonld,
        }

    except Exception as e:

        return {
            "ok": False,
            "url": url,
            "error": str(e),
        }


# ============================================================
# GEMINI RESEARCH
# ============================================================

def research_sources(
    country,
    region,
    category,
    target_indian_region,
):

    client = get_gemini_client()

    if not client:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    prompt = f"""
You are the global culinary research agent for
a restaurant R&D company.

Your job is to research real recipes on the public web.

RESEARCH TARGET

Country:
{country}

Region:
{region}

Food category:
{category}

Target Indian market:
{target_indian_region}

Find strong recipe sources that can be used as
research material for developing Indian street food.

Search broadly across the public web.

PRIORITIZE:

1. Regional food experts.
2. Established recipe publishers.
3. Chefs.
4. Food magazines.
5. Culinary institutions.
6. Local/regional food sites.
7. High-quality independent food writers.

Prefer recipes that have:

- actual ingredient lists
- measurable quantities
- clear preparation
- regional context
- identifiable culinary tradition
- useful cooking techniques
- useful flavour structure
- potential for adaptation

AVOID:

- SEO spam
- recipe aggregators
- duplicate recipes
- pages without recipes
- obvious AI-generated content
- pages that only contain snippets
- inaccessible/login-only pages
- pages with no meaningful recipe information

Do not invent URLs.

Return real URLs discovered through web search.

The purpose is NOT to copy recipes verbatim.

We are using these sources for culinary research
and then creating an original Indian adaptation.

Find approximately 10-12 strong candidates.

For each candidate evaluate:

- source quality
- recipe completeness
- regional authenticity
- Indian adaptation potential
- why it is interesting

Be particularly interested in recipes that could become
street-food products.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    google_search=types.GoogleSearch()
                )
            ],
            response_mime_type="application/json",
            response_schema=ResearchResult,
            temperature=0.25,
        ),
    )

    if getattr(
        response,
        "parsed",
        None,
    ):

        return response.parsed

    return ResearchResult.model_validate_json(
        response.text
    )


# ============================================================
# RECIPE EXTRACTION
# ============================================================

def extract_recipe(
    page,
    source: ResearchSource,
):

    client = get_gemini_client()

    if not client:
        raise RuntimeError(
            "Gemini API key is not configured."
        )

    prompt = f"""
You are a professional culinary data extraction system.

Extract the recipe from the webpage below.

Do NOT adapt it.

Do NOT Indianize it.

Do NOT invent ingredients.

Preserve quantities as closely as possible.

If a quantity is unknown, leave it blank.

SOURCE INFORMATION

Title:
{source.title}

URL:
{source.url}

Country:
{source.country}

Region:
{source.region}

Cuisine:
{source.cuisine}

JSON-LD DATA:
{json.dumps(page.get("jsonld", []), ensure_ascii=False)[:12000]}

PAGE TEXT:
{page.get("text", "")[:18000]}

Return a clean structured recipe.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Recipe,
            temperature=0.15,
        ),
    )

    if getattr(
        response,
        "parsed",
        None,
    ):

        recipe = response.parsed

    else:

        recipe = Recipe.model_validate_json(
            response.text
        )

    recipe.source_url = (
        page.get("url")
        or source.url
    )

    recipe.source_title = (
        source.title
    )

    recipe.country = (
        recipe.country
        or source.country
    )

    recipe.region = (
        recipe.region
        or source.region
    )

    recipe.cuisine = (
        recipe.cuisine
        or source.cuisine
    )

    return recipe


# ============================================================
# DEDUPLICATION
# ============================================================

def recipe_fingerprint(recipe):

    ingredients = "|".join(
        sorted(
            i.name.lower().strip()
            for i in recipe.ingredients
        )
    )

    base = (
        recipe.title.lower().strip()
        + "|"
        + ingredients
    )

    return hashlib.sha256(
        base.encode("utf-8")
    ).hexdigest()


def deduplicate_recipes(
    recipes,
):

    output = []
    seen = set()

    for recipe in recipes:

        fp = recipe_fingerprint(
            recipe
        )

        if fp not in seen:

            seen.add(fp)
            output.append(recipe)

    return output


# ============================================================
# INDIAN ADAPTATION
# ============================================================

def adapt_recipes(
    recipes,
    target_indian_region,
    flavour_profile,
    spice,
    tang,
    sweet,
    salt,
    richness,
    smoke,
    vegetarian,
    vegan,
    jain,
    street_food,
):

    client = get_gemini_client()

    if not client:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    recipes_json = json.dumps(
        [
            r.model_dump()
            for r in recipes
        ],
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are an expert Indian chef,
food scientist,
street-food entrepreneur,
menu engineer,
and restaurant R&D consultant.

We researched international recipes.

Now transform them into original Indian
street-food products.

TARGET INDIAN REGION:
{target_indian_region}

FLAVOUR PROFILE:
{flavour_profile}

FLAVOUR CONTROLS:

Spice: {spice}/10
Tanginess: {tang}/10
Sweetness: {sweet}/10
Saltiness: {salt}/10
Richness: {richness}/10
Smokiness: {smoke}/10

DIETARY REQUIREMENTS:

Vegetarian:
{vegetarian}

Vegan:
{vegan}

Jain:
{jain}

STREET FOOD:
{street_food}

SOURCE RECIPES:

{recipes_json}

CULINARY INSTRUCTIONS:

Do not merely add chilli powder.

Think like a professional Indian chef.

Consider:

- Indian flavour architecture
- regional Indian cooking
- ingredient availability
- texture
- aroma
- spice
- acidity
- sweetness
- chutneys
- masalas
- frying
- tawa cooking
- grilling
- smoking
- fermentation where appropriate
- Indian street-food formats
- speed of service
- batch preparation
- holding quality
- consistency
- portion size
- food cost
- customer appeal

PRESERVE THE CORE IDENTITY:

The adapted product should still retain
something recognizable from the original recipe.

However, it may be reformatted as:

- roll
- wrap
- toast
- sandwich
- chaat
- tikka
- skewer
- fried snack
- tawa snack
- stuffed bread
- bowl
- burger
- dosa-style product
- Indian fusion street-food item

if that makes commercial sense.

IMPORTANT:

Do not claim authenticity when the adaptation
is clearly fusion.

Create original recipe wording.

Provide practical quantities.

Provide a realistic preparation method.

Estimate food cost in INR for one serving.

Suggest a realistic Indian street-food selling price.

Score:

street_food_score:
0-10

commercial_score:
0-10

Also explain why the product could work
or fail commercially.

Return the strongest adapted concepts.
"""


    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GenerationResult,
            temperature=0.65,
        ),
    )

    if getattr(
        response,
        "parsed",
        None,
    ):

        result = response.parsed

    else:

        result = GenerationResult.model_validate_json(
            response.text
        )

    return result


# ============================================================
# RESEARCH PIPELINE
# ============================================================

def run_full_pipeline(
    country,
    region,
    category,
    target_indian_region,
    flavour_profile,
    spice,
    tang,
    sweet,
    salt,
    richness,
    smoke,
    vegetarian,
    vegan,
    jain,
    street_food,
):

    progress = st.progress(
        0
    )

    status = st.empty()

    # --------------------------------------------------------
    # 1. WEB RESEARCH
    # --------------------------------------------------------

    status.info(
        "🌍 Step 1/5 — Gemini is researching the global web..."
    )

    progress.progress(
        10
    )

    research = research_sources(
        country=country,
        region=region,
        category=category,
        target_indian_region=target_indian_region,
    )

    sources = []

    seen_urls = set()

    for source in research.sources:

        url = normalize_url(
            source.url
        )

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        sources.append(
            source
        )

    sources.sort(
        key=lambda x: (
            x.source_quality
            + x.recipe_completeness
            + x.regional_authenticity
            + x.indian_adaptation_potential
        ),
        reverse=True,
    )

    sources = sources[
        :MAX_RESEARCH_SOURCES
    ]

    # --------------------------------------------------------
    # 2. FETCH
    # --------------------------------------------------------

    status.info(
        f"🌐 Step 2/5 — Checking {len(sources)} recipe sources..."
    )

    progress.progress(
        30
    )

    fetched_pages = []

    for i, source in enumerate(
        sources
    ):

        page = fetch_webpage(
            source.url
        )

        if page.get("ok"):

            fetched_pages.append(
                (
                    source,
                    page,
                )
            )

        progress.progress(
            30
            + int(
                (i + 1)
                / max(len(sources), 1)
                * 20
            )
        )

    # --------------------------------------------------------
    # 3. EXTRACT RECIPES
    # --------------------------------------------------------

    status.info(
        f"🍳 Step 3/5 — Extracting {len(fetched_pages)} recipes..."
    )

    recipes = []

    for i, (
        source,
        page,
    ) in enumerate(
        fetched_pages[
            :MAX_RECIPES_TO_EXTRACT
        ]
    ):

        try:

            recipe = extract_recipe(
                page,
                source,
            )

            if recipe.ingredients:

                recipes.append(
                    recipe
                )

        except Exception:
            pass

        progress.progress(
            50
            + int(
                (i + 1)
                / max(
                    min(
                        len(fetched_pages),
                        MAX_RECIPES_TO_EXTRACT,
                    ),
                    1,
                )
                * 20
            )
        )

        time.sleep(
            0.2
        )

    recipes = deduplicate_recipes(
        recipes
    )

    if not recipes:

        raise RuntimeError(
            "No usable recipes could be extracted "
            "from the discovered sources."
        )

    # --------------------------------------------------------
    # 4. INDIANIZATION
    # --------------------------------------------------------

    status.info(
        f"🇮🇳 Step 4/5 — Creating Indian adaptations..."
    )

    progress.progress(
        75
    )

    adapted = adapt_recipes(
        recipes=recipes,
        target_indian_region=target_indian_region,
        flavour_profile=flavour_profile,
        spice=spice,
        tang=tang,
        sweet=sweet,
        salt=salt,
        richness=richness,
        smoke=smoke,
        vegetarian=vegetarian,
        vegan=vegan,
        jain=jain,
        street_food=street_food,
    )

    # --------------------------------------------------------
    # 5. COMPLETE
    # --------------------------------------------------------

    progress.progress(
        100
    )

    status.success(
        f"✅ Research complete — "
        f"{len(adapted.recipes)} Indian concepts created."
    )

    return (
        research,
        recipes,
        adapted,
    )


# ============================================================
# SCORE
# ============================================================

def business_score(
    recipe: AdaptedRecipe
):

    return round(
        (
            recipe.street_food_score
            * 0.5
            +
            recipe.commercial_score
            * 0.5
        )
        * 10,
        1,
    )


# ============================================================
# SESSION STATE
# ============================================================

if "last_research" not in st.session_state:
    st.session_state.last_research = None

if "last_original_recipes" not in st.session_state:
    st.session_state.last_original_recipes = []

if "last_adapted" not in st.session_state:
    st.session_state.last_adapted = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
<div class="hero">

<h1>🌶️ {APP_NAME}</h1>

<p>
Global culinary research → Indian flavour intelligence
→ street-food product development
</p>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# API CHECK
# ============================================================

if not gemini_ready():

    st.warning(
        """
        Gemini API is not configured.

        Add GEMINI_API_KEY to your environment or
        Streamlit secrets before running research.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌍 Research")

    country = st.selectbox(
        "Country",
        [
            "Mexico",
            "Thailand",
            "Japan",
            "South Korea",
            "Vietnam",
            "Indonesia",
            "Malaysia",
            "China",
            "Turkey",
            "Lebanon",
            "Greece",
            "Italy",
            "Spain",
            "Portugal",
            "France",
            "United Kingdom",
            "United States",
            "Brazil",
            "Argentina",
            "Peru",
            "Colombia",
            "South Africa",
            "Morocco",
            "Egypt",
            "Nigeria",
            "Ethiopia",
            "Australia",
            "New Zealand",
            "Other",
        ],
    )

    region = st.text_input(
        "Region / city / culinary area",
        placeholder="e.g. Oaxaca",
        value="",
    )

    category = st.selectbox(
        "Food category",
        [
            "Street Food",
            "Snacks",
            "Breakfast",
            "Sandwiches",
            "Wraps",
            "Fried Food",
            "Grilled Food",
            "Noodles",
            "Rice Dishes",
            "Dumplings",
            "Tacos / Flatbreads",
            "Desserts",
            "Drinks",
            "Sauces / Chutneys",
            "Any",
        ],
    )

    st.divider()

    st.header("🇮🇳 Indian Target")

    target_indian_region = st.selectbox(
        "Indian region",
        [
            "Pan-India",
            "Mumbai / Maharashtra",
            "Delhi / NCR",
            "Punjab",
            "Rajasthan",
            "Gujarat",
            "Karnataka",
            "Bengaluru",
            "Tamil Nadu",
            "Kerala",
            "Andhra Pradesh",
            "Telangana",
            "West Bengal",
            "Odisha",
            "Goa",
            "Northeast India",
        ],
    )

    flavour_profile = st.selectbox(
        "Flavour profile",
        [
            "Classic Indian Street Food",
            "Chatpata",
            "Spicy & Tangy",
            "Mumbai Street Food",
            "Delhi Street Food",
            "South Indian",
            "Punjabi",
            "Tandoori / Smoky",
            "Indo-Chinese",
            "Sweet & Spicy",
            "Modern Indian Fusion",
            "Premium Gourmet Street Food",
        ],
    )

    st.divider()

    st.header("🎚️ Flavour Controls")

    spice = st.slider(
        "🌶️ Spice",
        0,
        10,
        6,
    )

    tang = st.slider(
        "🍋 Tanginess",
        0,
        10,
        7,
    )

    sweet = st.slider(
        "🍬 Sweetness",
        0,
        10,
        3,
    )

    salt = st.slider(
        "🧂 Saltiness",
        0,
        10,
        6,
    )

    richness = st.slider(
        "🥛 Richness",
        0,
        10,
        5,
    )

    smoke = st.slider(
        "🔥 Smokiness",
        0,
        10,
        5,
    )

    st.divider()

    st.header("🥬 Dietary")

    vegetarian = st.checkbox(
        "Vegetarian",
        value=True,
    )

    vegan = st.checkbox(
        "Vegan",
        value=False,
    )

    jain = st.checkbox(
        "Jain-friendly",
        value=False,
    )

    street_food = st.checkbox(
        "Optimize for street food",
        value=True,
    )


# ============================================================
# MAIN GENERATE BUTTON
# ============================================================

st.subheader(
    "🚀 One-click culinary research"
)

st.write(
    """
Select your origin cuisine and Indian target above.
MasalaAtlas will research the web, select strong recipes,
extract them, and create Indian street-food adaptations.
"""
)

generate = st.button(
    "🔎 RESEARCH & GENERATE RECIPES",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN
# ============================================================

if generate:

    if not gemini_ready():

        st.error(
            "Gemini API key is missing."
        )

    elif not region.strip():

        st.error(
            "Please enter a region, city or culinary area."
        )

    else:

        try:

            (
                research,
                originals,
                adapted,
            ) = run_full_pipeline(
                country=country,
                region=region,
                category=category,
                target_indian_region=target_indian_region,
                flavour_profile=flavour_profile,
                spice=spice,
                tang=tang,
                sweet=sweet,
                salt=salt,
                richness=richness,
                smoke=smoke,
                vegetarian=vegetarian,
                vegan=vegan,
                jain=jain,
                street_food=street_food,
            )

            st.session_state.last_research = research

            st.session_state.last_original_recipes = originals

            st.session_state.last_adapted = adapted

        except Exception as e:

            st.error(
                "The research pipeline encountered an error."
            )

            with st.expander(
                "Technical error"
            ):
                st.exception(e)


# ============================================================
# RESULTS
# ============================================================

if st.session_state.last_adapted:

    adapted_result = (
        st.session_state.last_adapted
    )

    research_result = (
        st.session_state.last_research
    )

    original_recipes = (
        st.session_state.last_original_recipes
    )

    st.divider()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.header(
        "📊 Research Summary"
    )

    st.markdown(
        f"""
<div class="success-box">

<strong>Research target:</strong>
{country} → {region}

<br><br>

<strong>Indian target:</strong>
{target_indian_region}

<br><br>

<strong>Recipes researched:</strong>
{len(research_result.sources)}

<br>

<strong>Recipes successfully extracted:</strong>
{len(original_recipes)}

<br>

<strong>Indian concepts created:</strong>
{len(adapted_result.recipes)}

</div>
""",
        unsafe_allow_html=True,
    )

    st.write(
        adapted_result.research_summary
    )

    # --------------------------------------------------------
    # RESEARCH SOURCES
    # --------------------------------------------------------

    with st.expander(
        "🌐 View AI research sources"
    ):

        for source in research_result.sources:

            total_score = (
                source.source_quality
                + source.recipe_completeness
                + source.regional_authenticity
                + source.indian_adaptation_potential
            )

            st.markdown(
                f"""
<div class="source-card">

<h4>{source.title}</h4>

<strong>Source:</strong>
<a href="{source.url}" target="_blank">
{source.url}
</a>

<br><br>

<strong>Quality:</strong>
{source.source_quality}/10

&nbsp;&nbsp;

<strong>Completeness:</strong>
{source.recipe_completeness}/10

&nbsp;&nbsp;

<strong>Regional:</strong>
{source.regional_authenticity}/10

&nbsp;&nbsp;

<strong>Indian potential:</strong>
{source.indian_adaptation_potential}/10

<br><br>

{source.why_relevant}

</div>
""",
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # BUSINESS DASHBOARD
    # --------------------------------------------------------

    st.header(
        "🏪 Restaurant R&D Dashboard"
    )

    if adapted_result.recipes:

        scores = [
            business_score(r)
            for r in adapted_result.recipes
        ]

        avg_score = (
            sum(scores)
            / len(scores)
        )

        best_score = max(scores)

        avg_costs = [
            r.estimated_food_cost_inr
            for r in adapted_result.recipes
            if r.estimated_food_cost_inr
        ]

        avg_cost = (
            sum(avg_costs)
            / len(avg_costs)
            if avg_costs
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Concepts",
            len(adapted_result.recipes),
        )

        c2.metric(
            "Avg business score",
            f"{avg_score:.0f}/100",
        )

        c3.metric(
            "Best score",
            f"{best_score:.0f}/100",
        )

        c4.metric(
            "Avg food cost",
            f"₹{avg_cost:.0f}",
        )

    # --------------------------------------------------------
    # SORT RECIPES
    # --------------------------------------------------------

    ranked = sorted(
        adapted_result.recipes,
        key=business_score,
        reverse=True,
    )

    st.header(
        "🍽️ Indian Street-Food Concepts"
    )

    # --------------------------------------------------------
    # RECIPE CARDS
    # --------------------------------------------------------

    for index, recipe in enumerate(
        ranked,
        1,
    ):

        score = business_score(
            recipe
        )

        with st.container():

            st.markdown(
                '<div class="recipe-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
## {index}. {recipe.indian_name}

**Original:** {recipe.original_title}

**Source:**
<a href="{recipe.source_url}" target="_blank">
Open original source
</a>
""",
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Business",
                f"{score:.0f}/100",
            )

            c2.metric(
                "Street Food",
                f"{recipe.street_food_score}/10",
            )

            c3.metric(
                "Commercial",
                f"{recipe.commercial_score}/10",
            )

            c4.metric(
                "Food Cost",
                f"₹{recipe.estimated_food_cost_inr:.0f}",
            )

            c5.metric(
                "Suggested Price",
                f"₹{recipe.suggested_selling_price_inr:.0f}",
            )

            st.markdown(
                f"""
**Indian positioning**

{recipe.menu_positioning}

**Description**

{recipe.description}
"""
            )

            st.markdown(
                "### 🎚️ Flavour fingerprint"
            )

            flavour_cols = st.columns(
                6
            )

            flavour_values = [
                (
                    "🌶️",
                    "Spice",
                    recipe.spice_level,
                ),
                (
                    "🍋",
                    "Tang",
                    recipe.tanginess,
                ),
                (
                    "🍬",
                    "Sweet",
                    recipe.sweetness,
                ),
                (
                    "🧂",
                    "Salt",
                    recipe.saltiness,
                ),
                (
                    "🥛",
                    "Rich",
                    recipe.richness,
                ),
                (
                    "🔥",
                    "Smoke",
                    recipe.smokiness,
                ),
            ]

            for col, (
                icon,
                label,
                value,
            ) in zip(
                flavour_cols,
                flavour_values,
            ):

                with col:

                    st.metric(
                        f"{icon} {label}",
                        f"{value}/10",
                    )

            st.markdown(
                "### 🧂 Ingredients"
            )

            for ingredient in (
                recipe.ingredients
            ):

                notes = ""

                if ingredient.notes:
                    notes = (
                        f" — {ingredient.notes}"
                    )

                st.write(
                    f"• **{ingredient.quantity}** "
                    f"{ingredient.name}"
                    f"{notes}"
                )

            st.markdown(
                "### 👨‍🍳 Preparation"
            )

            for i, step in enumerate(
                recipe.instructions,
                1,
            ):

                st.write(
                    f"**{i}.** {step}"
                )

            if recipe.accompaniments:

                st.markdown(
                    "### 🥣 Accompaniments"
                )

                for item in (
                    recipe.accompaniments
                ):

                    st.write(
                        f"• {item}"
                    )

            if recipe.substitutions:

                st.markdown(
                    "### 🔄 Indian substitutions"
                )

                for item in (
                    recipe.substitutions
                ):

                    st.write(
                        f"• {item}"
                    )

            if recipe.adaptation_notes:

                st.markdown(
                    "### 🇮🇳 Why this works for the Indian palate"
                )

                for item in (
                    recipe.adaptation_notes
                ):

                    st.write(
                        f"• {item}"
                    )

            if recipe.batch_production_notes:

                st.markdown(
                    "### 🏪 Restaurant batch production"
                )

                for item in (
                    recipe.batch_production_notes
                ):

                    st.write(
                        f"• {item}"
                    )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    st.divider()

    st.header(
        "💾 Export R&D Results"
    )

    export_data = json.dumps(
        [
            recipe.model_dump()
            for recipe in ranked
        ],
        ensure_ascii=False,
        indent=2,
    )

    st.download_button(
        label="⬇️ Download all generated recipes",
        data=export_data,
        file_name="masalaatlas_rnd_results.json",
        mime="application/json",
        use_container_width=True,
    )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.divider()

    st.markdown(
        """
### How it works

**1. Choose the food world**

Pick a country, region and food category.

**2. Choose your Indian target**

Tell the AI which Indian market or regional palate
you want to target.

**3. Tune the flavour**

Adjust spice, tang, sweetness, salt, richness
and smokiness.

**4. Press one button**

The AI researches the web, finds promising recipes,
extracts them and creates Indian street-food concepts.

**5. Evaluate the business**

Each concept receives a street-food score,
commercial score, estimated food cost and
suggested selling price.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
MasalaAtlas AI · Global culinary research and Indian
street-food R&D.

Recipe sources are used for research. Respect the
source websites' terms, robots policies, copyright,
licensing requirements and applicable law. The app
does not bypass authentication, paywalls or access controls.
"""
)
