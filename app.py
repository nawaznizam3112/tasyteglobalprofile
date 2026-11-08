import os
import re
import json
import time
import hashlib
from typing import Any, Optional

import requests
import streamlit as st
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "Tasyte Global Recipe R&D"
APP_VERSION = "2.0"

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

MAX_SEARCH_RESULTS = 8
MAX_FETCHED_RECIPES = 6
MAX_FINAL_RECIPES = 5

REQUEST_TIMEOUT = 18
CACHE_TTL = 60 * 60 * 24

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            rgba(255, 153, 51, 0.18),
            rgba(255, 255, 255, 0.05)
        );
        border: 1px solid rgba(255, 153, 51, 0.25);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.3rem;
    }

    .hero p {
        margin-top: 0;
        opacity: 0.8;
    }

    .recipe-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        background: rgba(128,128,128,0.04);
    }

    .score {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .small {
        font-size: 0.85rem;
        opacity: 0.7;
    }

    .tag {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        margin: 0.15rem;
        border-radius: 999px;
        background: rgba(255, 153, 51, 0.15);
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODELS
# ============================================================

class ResearchSource(BaseModel):
    title: str = ""
    url: str = ""
    reason: str = ""


class Ingredient(BaseModel):
    name: str
    quantity: str = ""
    notes: str = ""


class Recipe(BaseModel):
    title: str
    source_url: str = ""
    source_name: str = ""
    description: str = ""
    servings: str = ""
    prep_time: str = ""
    cook_time: str = ""
    ingredients: list[Ingredient] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    cuisine: str = ""
    category: str = ""


class AdaptedRecipe(BaseModel):
    original_title: str
    final_name: str

    positioning: str = ""

    target_region: str = ""
    flavour_profile: dict[str, float] = Field(default_factory=dict)

    ingredients: list[Ingredient] = Field(default_factory=list)
    method: list[str] = Field(default_factory=list)

    serving_style: str = ""
    garnish: str = ""
    chutney_or_sauce: str = ""

    street_food_notes: list[str] = Field(default_factory=list)
    operational_notes: list[str] = Field(default_factory=list)

    vegetarian: bool = True
    vegan: bool = False
    jain_friendly: bool = False

    estimated_food_cost_inr: float = 0
    suggested_selling_price_inr: float = 0

    commercial_score: float = 0
    novelty_score: float = 0
    indian_fit_score: float = 0
    operational_score: float = 0

    why_it_can_work: str = ""
    risks: list[str] = Field(default_factory=list)

    source_url: str = ""


# ============================================================
# GEMINI
# ============================================================

class GeminiQuotaError(Exception):
    pass


class GeminiAPIError(Exception):
    pass


@st.cache_resource
def get_gemini_client():
    api_key = None

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise GeminiAPIError(
            "GEMINI_API_KEY is not configured. "
            "Add it to Streamlit secrets or environment variables."
        )

    return genai.Client(api_key=api_key)


def get_model_name():
    try:
        return st.secrets.get(
            "GEMINI_MODEL",
            DEFAULT_GEMINI_MODEL
        )
    except Exception:
        return os.getenv(
            "GEMINI_MODEL",
            DEFAULT_GEMINI_MODEL
        )


def is_quota_error(exc: Exception) -> bool:
    text = str(exc).upper()

    return any(
        token in text
        for token in [
            "429",
            "RESOURCE_EXHAUSTED",
            "QUOTA",
            "RATE LIMIT",
            "RATE_LIMIT",
            "TOO MANY REQUESTS",
        ]
    )


def call_gemini(
    prompt: str,
    *,
    use_google_search: bool = False,
    temperature: float = 0.2,
) -> Any:

    client = get_gemini_client()
    model = get_model_name()

    config_kwargs = {
        "temperature": temperature,
    }

    if use_google_search:
        config_kwargs["tools"] = [
            types.Tool(
                google_search=types.GoogleSearch()
            )
        ]

    config = types.GenerateContentConfig(
        **config_kwargs
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        return response

    except Exception as exc:

        if is_quota_error(exc):
            raise GeminiQuotaError(
                "Gemini quota/rate limit was reached."
            ) from exc

        raise GeminiAPIError(
            f"Gemini API error: {exc}"
        ) from exc


# ============================================================
# HELPERS
# ============================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(
            clean_text(x)
            for x in value
            if clean_text(x)
        )

    return re.sub(
        r"\s+",
        " ",
        str(value)
    ).strip()


def safe_json_loads(text: str) -> Any:
    text = text.strip()

    # Remove markdown fences
    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```$",
        "",
        text
    )

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Try finding first JSON object
    start_obj = text.find("{")
    end_obj = text.rfind("}")

    if start_obj >= 0 and end_obj > start_obj:
        candidate = text[start_obj:end_obj + 1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try finding JSON array
    start_arr = text.find("[")
    end_arr = text.rfind("]")

    if start_arr >= 0 and end_arr > start_arr:
        candidate = text[start_arr:end_arr + 1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Could not parse JSON from Gemini response.")


def normalize_url(url: str) -> str:
    if not url:
        return ""

    url = url.strip()

    url = re.sub(
        r"[?#].*$",
        "",
        url
    )

    return url.rstrip("/")


def domain_from_url(url: str) -> str:
    match = re.search(
        r"https?://(?:www\.)?([^/]+)",
        url or ""
    )

    return match.group(1) if match else ""


def parse_number(value: Any) -> float:
    if value is None:
        return 0.0

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        str(value)
    )

    if not match:
        return 0.0

    try:
        return float(match.group(0))
    except Exception:
        return 0.0


def parse_duration(value: Any) -> str:
    if not value:
        return ""

    value = str(value)

    # ISO 8601
    match = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?",
        value
    )

    if match:
        hours = match.group(1)
        minutes = match.group(2)

        parts = []

        if hours:
            parts.append(f"{hours} hr")

        if minutes:
            parts.append(f"{minutes} min")

        return " ".join(parts)

    return value


# ============================================================
# WEB FETCHING
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_webpage(url: str) -> Optional[str]:

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            return None

        content_type = response.headers.get(
            "content-type",
            ""
        ).lower()

        if (
            "text/html" not in content_type
            and "application/xhtml" not in content_type
        ):
            return None

        return response.text

    except Exception:
        return None


# ============================================================
# SCHEMA.ORG RECIPE EXTRACTION
# ============================================================

def find_recipe_objects(data: Any) -> list[dict]:

    found = []

    if isinstance(data, dict):

        obj_type = data.get("@type")

        if obj_type == "Recipe":
            found.append(data)

        elif isinstance(obj_type, list) and "Recipe" in obj_type:
            found.append(data)

        for value in data.values():
            found.extend(
                find_recipe_objects(value)
            )

    elif isinstance(data, list):

        for item in data:
            found.extend(
                find_recipe_objects(item)
            )

    return found


def extract_schema_recipe(html: str) -> Optional[dict]:

    soup = BeautifulSoup(
        html,
        "lxml"
    )

    scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    candidates = []

    for script in scripts:

        raw = script.string or script.get_text()

        if not raw.strip():
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        recipes = find_recipe_objects(data)

        candidates.extend(recipes)

    if not candidates:
        return None

    return candidates[0]


def schema_to_recipe(
    schema: dict,
    source_url: str
) -> Recipe:

    ingredients = []

    raw_ingredients = (
        schema.get("recipeIngredient")
        or []
    )

    for item in raw_ingredients:

        ingredients.append(
            Ingredient(
                name=clean_text(item)
            )
        )

    instructions = []

    raw_instructions = (
        schema.get("recipeInstructions")
        or []
    )

    for item in raw_instructions:

        if isinstance(item, str):

            text = clean_text(item)

            if text:
                instructions.append(text)

        elif isinstance(item, dict):

            text = clean_text(
                item.get("text")
                or item.get("name")
            )

            if text:
                instructions.append(text)

    return Recipe(
        title=clean_text(
            schema.get("name")
            or "Untitled Recipe"
        ),
        source_url=source_url,
        source_name=domain_from_url(source_url),
        description=clean_text(
            schema.get("description")
        ),
        servings=clean_text(
            schema.get("recipeYield")
        ),
        prep_time=parse_duration(
            schema.get("prepTime")
        ),
        cook_time=parse_duration(
            schema.get("cookTime")
        ),
        ingredients=ingredients,
        instructions=instructions,
        cuisine=clean_text(
            schema.get("recipeCuisine")
        ),
        category=clean_text(
            schema.get("recipeCategory")
        ),
    )


# ============================================================
# GEMINI FALLBACK RECIPE EXTRACTION
# ============================================================

def build_page_text(html: str) -> str:

    soup = BeautifulSoup(
        html,
        "lxml"
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
        ]
    ):
        tag.decompose()

    text = soup.get_text(
        separator="\n"
    )

    lines = []

    for line in text.splitlines():

        line = clean_text(line)

        if line:
            lines.append(line)

    text = "\n".join(lines)

    # Keep prompt reasonably small
    return text[:18000]


def extract_recipes_with_gemini(
    pages: list[dict]
) -> list[Recipe]:

    if not pages:
        return []

    payload = []

    for page in pages:

        payload.append(
            {
                "url": page["url"],
                "text": page["text"],
            }
        )

    prompt = f"""
You are a professional recipe data extraction engine.

Extract recipes from the supplied web pages.

IMPORTANT:
- Do not invent ingredients.
- Do not invent quantities.
- Preserve the original recipe as closely as possible.
- Ignore advertisements and unrelated page content.
- Return ONLY valid JSON.
- Return one object per successfully extracted recipe.

JSON format:

{{
  "recipes": [
    {{
      "title": "",
      "source_url": "",
      "source_name": "",
      "description": "",
      "servings": "",
      "prep_time": "",
      "cook_time": "",
      "ingredients": [
        {{
          "name": "",
          "quantity": "",
          "notes": ""
        }}
      ],
      "instructions": [],
      "cuisine": "",
      "category": ""
    }}
  ]
}}

WEB PAGES:

{json.dumps(payload, ensure_ascii=False)}
"""

    response = call_gemini(
        prompt,
        use_google_search=False,
        temperature=0.1,
    )

    data = safe_json_loads(
        response.text
    )

    recipes = []

    for item in data.get("recipes", []):

        try:
            recipes.append(
                Recipe.model_validate(item)
            )
        except Exception:
            continue

    return recipes


# ============================================================
# GOOGLE SEARCH GROUNDING
# ============================================================

def extract_grounding_sources(response) -> list[ResearchSource]:

    sources = []

    try:
        candidates = response.candidates or []

        for candidate in candidates:

            metadata = getattr(
                candidate,
                "grounding_metadata",
                None
            )

            if not metadata:
                continue

            chunks = getattr(
                metadata,
                "grounding_chunks",
                None
            ) or []

            for chunk in chunks:

                web = getattr(
                    chunk,
                    "web",
                    None
                )

                if not web:
                    continue

                url = getattr(
                    web,
                    "uri",
                    ""
                )

                title = getattr(
                    web,
                    "title",
                    ""
                )

                if url:

                    sources.append(
                        ResearchSource(
                            title=title or domain_from_url(url),
                            url=normalize_url(url),
                            reason="Found through Google Search grounding",
                        )
                    )

    except Exception:
        pass

    return sources


def deduplicate_sources(
    sources: list[ResearchSource]
) -> list[ResearchSource]:

    seen = set()
    result = []

    for source in sources:

        url = normalize_url(
            source.url
        )

        if not url:
            continue

        key = url.lower()

        if key in seen:
            continue

        seen.add(key)

        result.append(
            ResearchSource(
                title=source.title,
                url=url,
                reason=source.reason,
            )
        )

    return result


def research_sources(
    country: str,
    region: str,
    category: str,
    target_region: str,
) -> list[ResearchSource]:

    prompt = f"""
You are the global culinary research assistant for a restaurant R&D team.

Find high-quality ORIGINAL recipes from the internet that match:

Country: {country}
Region: {region}
Food category: {category}

The recipes will later be adapted for:
Indian target region: {target_region}

Search broadly across:
- local food blogs
- regional culinary websites
- respected recipe publishers
- traditional food sites
- chef-authored recipes
- cultural food resources

Prioritize recipes that are:
1. Authentic to the selected country/region.
2. Clearly documented.
3. Practical enough to reproduce.
4. Interesting for restaurant R&D.
5. Not duplicate versions of the same recipe.

Do NOT focus on Indian recipes yet.
We need the ORIGINAL global recipes first.

Return a concise research report identifying the strongest recipe pages.

Do not invent URLs.
Use the web search tool to find real pages.
"""

    response = call_gemini(
        prompt,
        use_google_search=True,
        temperature=0.1,
    )

    sources = extract_grounding_sources(
        response
    )

    # Also inspect text for URLs if grounding metadata
    # is unavailable.
    if not sources:

        urls = re.findall(
            r"https?://[^\s\]\)\"<>]+",
            response.text or ""
        )

        for url in urls:

            sources.append(
                ResearchSource(
                    title=domain_from_url(url),
                    url=normalize_url(url),
                    reason="URL identified in research response",
                )
            )

    sources = deduplicate_sources(
        sources
    )

    return sources[:MAX_SEARCH_RESULTS]


# ============================================================
# RECIPE COLLECTION
# ============================================================

def collect_recipes(
    sources: list[ResearchSource]
) -> tuple[list[Recipe], list[ResearchSource]]:

    recipes = []
    pages_needing_ai = []

    used_urls = set()

    for source in sources:

        if len(recipes) >= MAX_FETCHED_RECIPES:
            break

        url = normalize_url(
            source.url
        )

        if not url or url in used_urls:
            continue

        used_urls.add(url)

        html = fetch_webpage(url)

        if not html:
            continue

        schema = extract_schema_recipe(
            html
        )

        if schema:

            try:
                recipe = schema_to_recipe(
                    schema,
                    url
                )

                if recipe.ingredients and recipe.instructions:
                    recipes.append(recipe)
                    continue

            except Exception:
                pass

        # Fallback:
        # Keep page text for ONE batch Gemini call.
        pages_needing_ai.append(
            {
                "url": url,
                "text": build_page_text(html),
            }
        )

    # AI fallback only when necessary.
    remaining_slots = (
        MAX_FETCHED_RECIPES - len(recipes)
    )

    if (
        pages_needing_ai
        and remaining_slots > 0
    ):

        pages_needing_ai = pages_needing_ai[
            :remaining_slots
        ]

        try:

            ai_recipes = extract_recipes_with_gemini(
                pages_needing_ai
            )

            recipes.extend(
                ai_recipes
            )

        except GeminiQuotaError:
            # We can still continue with schema recipes.
            pass

        except Exception:
            pass

    # Deduplicate recipes
    final_recipes = []

    seen_titles = set()

    for recipe in recipes:

        key = re.sub(
            r"[^a-z0-9]+",
            "",
            recipe.title.lower()
        )

        if key in seen_titles:
            continue

        seen_titles.add(key)

        final_recipes.append(
            recipe
        )

    return (
        final_recipes[:MAX_FETCHED_RECIPES],
        sources,
    )


# ============================================================
# FLAVOUR PROFILE
# ============================================================

def build_flavour_profile(
    spice: int,
    tanginess: int,
    sweetness: int,
    saltiness: int,
    richness: int,
    smokiness: int,
) -> dict[str, float]:

    return {
        "spice": spice,
        "tanginess": tanginess,
        "sweetness": sweetness,
        "saltiness": saltiness,
        "richness": richness,
        "smokiness": smokiness,
    }


# ============================================================
# ADAPTATION
# ============================================================

def adapt_recipes(
    recipes: list[Recipe],
    target_region: str,
    flavour_profile: dict[str, float],
    vegetarian: bool,
    vegan: bool,
    jain: bool,
) -> list[AdaptedRecipe]:

    if not recipes:
        return []

    recipes_payload = [
        recipe.model_dump()
        for recipe in recipes
    ]

    dietary_rules = []

    if vegetarian:
        dietary_rules.append(
            "All final recipes must be vegetarian."
        )

    if vegan:
        dietary_rules.append(
            "All final recipes must be vegan."
        )

    if jain:
        dietary_rules.append(
            "Avoid onion, garlic, root vegetables and other ingredients that conflict with Jain preparation."
        )

    if not dietary_rules:
        dietary_rules.append(
            "No mandatory vegetarian/vegan/Jain restriction."
        )

    prompt = f"""
You are the executive chef and restaurant R&D strategist
for a modern Indian street-food restaurant.

We have researched ORIGINAL recipes from different parts
of the world.

Your job is NOT to simply translate them.

You must intelligently adapt each recipe for:

TARGET INDIAN REGION:
{target_region}

FLAVOUR PROFILE:
{json.dumps(flavour_profile, indent=2)}

DIETARY REQUIREMENTS:
{chr(10).join("- " + x for x in dietary_rules)}

The result should feel commercially viable for Indian customers
while retaining a recognizable identity from the original recipe.

IMPORTANT:

1. Preserve the soul of the original dish.
2. Do not make every recipe taste the same.
3. Use Indian ingredients intelligently.
4. Consider ingredient availability in India.
5. Consider street-food service.
6. Consider batch preparation.
7. Consider holding time.
8. Consider speed of service.
9. Consider food cost.
10. Consider customer familiarity versus novelty.
11. Avoid unnecessary expensive ingredients.
12. Give practical quantities.
13. Make the recipe operationally realistic.
14. Do not claim something is authentic if it is an adaptation.

Return ONLY valid JSON.

JSON format:

{{
  "recipes": [
    {{
      "original_title": "",
      "final_name": "",
      "positioning": "",
      "target_region": "",
      "flavour_profile": {{
        "spice": 0,
        "tanginess": 0,
        "sweetness": 0,
        "saltiness": 0,
        "richness": 0,
        "smokiness": 0
      }},
      "ingredients": [
        {{
          "name": "",
          "quantity": "",
          "notes": ""
        }}
      ],
      "method": [],
      "serving_style": "",
      "garnish": "",
      "chutney_or_sauce": "",
      "street_food_notes": [],
      "operational_notes": [],
      "vegetarian": true,
      "vegan": false,
      "jain_friendly": false,
      "estimated_food_cost_inr": 0,
      "suggested_selling_price_inr": 0,
      "commercial_score": 0,
      "novelty_score": 0,
      "indian_fit_score": 0,
      "operational_score": 0,
      "why_it_can_work": "",
      "risks": [],
      "source_url": ""
    }}
  ]
}}

SOURCE RECIPES:

{json.dumps(recipes_payload, ensure_ascii=False, indent=2)}
"""

    response = call_gemini(
        prompt,
        use_google_search=False,
        temperature=0.35,
    )

    data = safe_json_loads(
        response.text
    )

    results = []

    for item in data.get("recipes", []):

        try:
            results.append(
                AdaptedRecipe.model_validate(
                    item
                )
            )

        except Exception:
            continue

    return results[:MAX_FINAL_RECIPES]


# ============================================================
# SIMPLE LOCAL SCORING
# ============================================================

def calculate_local_score(
    recipe: AdaptedRecipe
) -> float:

    values = [
        recipe.commercial_score,
        recipe.novelty_score,
        recipe.indian_fit_score,
        recipe.operational_score,
    ]

    valid = [
        max(0, min(10, float(v)))
        for v in values
    ]

    if not valid:
        return 0

    return round(
        sum(valid) / len(valid) * 10,
        1
    )


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    country: str,
    region: str,
    category: str,
    target_region: str,
    flavour_profile: dict[str, float],
    vegetarian: bool,
    vegan: bool,
    jain: bool,
):

    progress = st.progress(
        0,
        text="Starting research..."
    )

    status = st.empty()

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    status.info(
        "🌍 Searching the web for original recipes..."
    )

    progress.progress(
        15,
        text="Searching global recipes..."
    )

    sources = research_sources(
        country=country,
        region=region,
        category=category,
        target_region=target_region,
    )

    if not sources:
        raise RuntimeError(
            "No useful recipe sources were found."
        )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    status.info(
        f"🔎 Found {len(sources)} candidate sources. Fetching recipe pages..."
    )

    progress.progress(
        35,
        text="Reading recipe pages..."
    )

    recipes, sources = collect_recipes(
        sources
    )

    if not recipes:
        raise RuntimeError(
            "Search succeeded, but no recipes could be extracted from the discovered pages."
        )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    status.info(
        f"🍳 Extracted {len(recipes)} recipes. Building Indian adaptations..."
    )

    progress.progress(
        65,
        text="Adapting recipes..."
    )

    adapted = adapt_recipes(
        recipes=recipes,
        target_region=target_region,
        flavour_profile=flavour_profile,
        vegetarian=vegetarian,
        vegan=vegan,
        jain=jain,
    )

    if not adapted:
        raise RuntimeError(
            "Recipes were extracted, but the adaptation stage returned no results."
        )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    progress.progress(
        90,
        text="Calculating R&D scores..."
    )

    for recipe in adapted:

        recipe.commercial_score = max(
            0,
            min(
                10,
                recipe.commercial_score
            )
        )

        recipe.novelty_score = max(
            0,
            min(
                10,
                recipe.novelty_score
            )
        )

        recipe.indian_fit_score = max(
            0,
            min(
                10,
                recipe.indian_fit_score
            )
        )

        recipe.operational_score = max(
            0,
            min(
                10,
                recipe.operational_score
            )
        )

    adapted.sort(
        key=calculate_local_score,
        reverse=True
    )

    progress.progress(
        100,
        text="R&D generation complete."
    )

    status.success(
        f"Completed: {len(adapted)} restaurant R&D concepts generated."
    )

    return sources, recipes, adapted


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🍽️ R&D Controls")

    st.caption(
        "Select the market and culinary direction. "
        "The system handles web research automatically."
    )

    st.divider()

    country = st.selectbox(
        "Original Country",
        [
            "Italy",
            "France",
            "United States",
            "Mexico",
            "Turkey",
            "Japan",
            "South Korea",
            "Thailand",
            "Vietnam",
            "Indonesia",
            "China",
            "Spain",
            "Greece",
            "Lebanon",
            "Morocco",
            "United Kingdom",
            "Germany",
            "Brazil",
            "Argentina",
            "South Africa",
            "Other",
        ]
    )

    region = st.text_input(
        "Original Region / City",
        placeholder="e.g. Naples, Oaxaca, Istanbul"
    )

    category = st.selectbox(
        "Food Category",
        [
            "Street Food",
            "Snack",
            "Sandwich",
            "Wrap",
            "Burger",
            "Pizza",
            "Fried Food",
            "Noodles",
            "Rice",
            "Dumpling",
            "Pastry",
            "Dessert",
            "Breakfast",
            "Soup",
            "Main Course",
            "Sauce / Dip",
        ]
    )

    target_region = st.selectbox(
        "Indian Target Region",
        [
            "Pan India",
            "North India",
            "South India",
            "West India",
            "East India",
            "Northeast India",
            "Maharashtra",
            "Gujarat",
            "Punjab",
            "Delhi NCR",
            "Rajasthan",
            "Karnataka",
            "Tamil Nadu",
            "Kerala",
            "Andhra Pradesh / Telangana",
            "West Bengal",
        ]
    )

    st.divider()

    st.subheader("🌶️ Flavour Profile")

    spice = st.slider(
        "Spice",
        0,
        100,
        65
    )

    tanginess = st.slider(
        "Tanginess",
        0,
        100,
        55
    )

    sweetness = st.slider(
        "Sweetness",
        0,
        100,
        30
    )

    saltiness = st.slider(
        "Saltiness",
        0,
        100,
        60
    )

    richness = st.slider(
        "Richness",
        0,
        100,
        50
    )

    smokiness = st.slider(
        "Smokiness",
        0,
        100,
        40
    )

    st.divider()

    st.subheader("🥗 Dietary")

    vegetarian = st.checkbox(
        "Vegetarian",
        value=True
    )

    vegan = st.checkbox(
        "Vegan",
        value=False
    )

    jain = st.checkbox(
        "Jain-friendly",
        value=False
    )

    if vegan:
        vegetarian = True

    st.divider()

    st.caption(
        f"Gemini model: `{get_model_name()}`"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <h1>🌍 → 🇮🇳 {APP_TITLE}</h1>
        <p>
            Global recipe discovery → automatic recipe extraction →
            Indian adaptation → street-food R&D evaluation
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN ACTION
# ============================================================

flavour_profile = build_flavour_profile(
    spice=spice,
    tanginess=tanginess,
    sweetness=sweetness,
    saltiness=saltiness,
    richness=richness,
    smokiness=smokiness,
)

if st.button(
    "🚀 RESEARCH & GENERATE",
    type="primary",
    use_container_width=True,
):

    if not region.strip():

        st.error(
            "Please enter the original region/city."
        )

        st.stop()

    try:

        with st.spinner(
            "Running restaurant R&D pipeline..."
        ):

            (
                sources,
                original_recipes,
                final_recipes,
            ) = run_pipeline(
                country=country,
                region=region,
                category=category,
                target_region=target_region,
                flavour_profile=flavour_profile,
                vegetarian=vegetarian,
                vegan=vegan,
                jain=jain,
            )

        st.session_state["sources"] = sources
        st.session_state["original_recipes"] = original_recipes
        st.session_state["final_recipes"] = final_recipes
        st.session_state["research_inputs"] = {
            "country": country,
            "region": region,
            "category": category,
            "target_region": target_region,
            "flavour_profile": flavour_profile,
        }

    except GeminiQuotaError:

        st.error(
            "⚠️ Gemini quota/rate limit was reached."
        )

        st.warning(
            "The app stopped instead of repeatedly retrying the API. "
            "Check your Gemini project quota/billing and try again later."
        )

        st.info(
            "The current version intentionally uses far fewer Gemini calls "
            "than the previous version."
        )

        st.stop()

    except GeminiAPIError as exc:

        st.error(
            "Gemini API configuration/error."
        )

        st.code(
            str(exc)
        )

        st.stop()

    except Exception as exc:

        st.error(
            "The R&D pipeline could not complete."
        )

        st.exception(exc)

        st.stop()


# ============================================================
# RESULTS
# ============================================================

if "final_recipes" in st.session_state:

    final_recipes = st.session_state[
        "final_recipes"
    ]

    original_recipes = st.session_state[
        "original_recipes"
    ]

    sources = st.session_state[
        "sources"
    ]

    research_inputs = st.session_state[
        "research_inputs"
    ]

    st.divider()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sources",
            len(sources)
        )

    with col2:
        st.metric(
            "Recipes Found",
            len(original_recipes)
        )

    with col3:
        st.metric(
            "R&D Concepts",
            len(final_recipes)
        )

    with col4:

        scores = [
            calculate_local_score(recipe)
            for recipe in final_recipes
        ]

        avg_score = (
            round(sum(scores) / len(scores), 1)
            if scores
            else 0
        )

        st.metric(
            "Average R&D Score",
            f"{avg_score}/100"
        )

    st.divider()

    # --------------------------------------------------------
    # FLAVOUR PROFILE
    # --------------------------------------------------------

    st.subheader("🎯 Selected R&D Direction")

    fp = research_inputs[
        "flavour_profile"
    ]

    cols = st.columns(6)

    flavour_names = [
        "Spice",
        "Tanginess",
        "Sweetness",
        "Saltiness",
        "Richness",
        "Smokiness",
    ]

    for col, name in zip(
        cols,
        flavour_names
    ):

        key = name.lower()

        with col:

            st.metric(
                name,
                fp[key]
            )

    # --------------------------------------------------------
    # RECIPES
    # --------------------------------------------------------

    st.subheader(
        "🔥 Restaurant R&D Concepts"
    )

    for index, recipe in enumerate(
        final_recipes,
        start=1
    ):

        score = calculate_local_score(
            recipe
        )

        with st.container():

            st.markdown(
                f"""
                <div class="recipe-card">
                    <h2>
                        #{index} — {recipe.final_name}
                    </h2>

                    <p>
                        <strong>Original:</strong>
                        {recipe.original_title}
                    </p>

                    <p>
                        <strong>Positioning:</strong>
                        {recipe.positioning}
                    </p>

                    <div>
                        <span class="tag">
                            R&D Score {score}/100
                        </span>

                        <span class="tag">
                            India Fit {recipe.indian_fit_score}/10
                        </span>

                        <span class="tag">
                            Commercial {recipe.commercial_score}/10
                        </span>

                        <span class="tag">
                            Operations {recipe.operational_score}/10
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)

            with c1:

                st.markdown(
                    "### 🧂 Ingredients"
                )

                for ingredient in recipe.ingredients:

                    quantity = (
                        ingredient.quantity
                        if ingredient.quantity
                        else ""
                    )

                    notes = (
                        f" — {ingredient.notes}"
                        if ingredient.notes
                        else ""
                    )

                    st.write(
                        f"• **{ingredient.name}** "
                        f"{quantity}{notes}"
                    )

            with c2:

                st.markdown(
                    "### 👨‍🍳 Method"
                )

                for i, step in enumerate(
                    recipe.method,
                    start=1
                ):

                    st.write(
                        f"{i}. {step}"
                    )

            st.markdown(
                "### 🍽️ Serving & Street-Food Format"
            )

            st.write(
                recipe.serving_style
            )

            if recipe.garnish:

                st.write(
                    f"**Garnish:** {recipe.garnish}"
                )

            if recipe.chutney_or_sauce:

                st.write(
                    f"**Sauce / Chutney:** "
                    f"{recipe.chutney_or_sauce}"
                )

            if recipe.street_food_notes:

                st.markdown(
                    "### 🏪 Street-Food Notes"
                )

                for note in recipe.street_food_notes:

                    st.write(
                        f"• {note}"
                    )

            if recipe.operational_notes:

                st.markdown(
                    "### ⚙️ Operations"
                )

                for note in recipe.operational_notes:

                    st.write(
                        f"• {note}"
                    )

            price1, price2, price3 = st.columns(3)

            with price1:

                st.metric(
                    "Estimated Food Cost",
                    f"₹{recipe.estimated_food_cost_inr:.0f}"
                )

            with price2:

                st.metric(
                    "Suggested Selling Price",
                    f"₹{recipe.suggested_selling_price_inr:.0f}"
                )

            with price3:

                st.metric(
                    "Novelty",
                    f"{recipe.novelty_score:.1f}/10"
                )

            st.markdown(
                "### 💡 Why It Can Work"
            )

            st.write(
                recipe.why_it_can_work
            )

            if recipe.risks:

                st.markdown(
                    "### ⚠️ Risks"
                )

                for risk in recipe.risks:

                    st.write(
                        f"• {risk}"
                    )

            if recipe.source_url:

                st.markdown(
                    f"**Original source:** "
                    f"[Open recipe source]({recipe.source_url})"
                )

            st.divider()

    # --------------------------------------------------------
    # ORIGINAL RECIPES
    # --------------------------------------------------------

    with st.expander(
        "📚 View Original Global Recipes"
    ):

        for recipe in original_recipes:

            st.markdown(
                f"### {recipe.title}"
            )

            if recipe.description:

                st.write(
                    recipe.description
                )

            st.write(
                f"**Source:** "
                f"[{recipe.source_name}]({recipe.source_url})"
            )

            st.write(
                f"Ingredients: "
                f"{len(recipe.ingredients)}"
            )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    with st.expander(
        "🌐 Research Sources"
    ):

        for source in sources:

            title = (
                source.title
                or domain_from_url(source.url)
            )

            st.markdown(
                f"- [{title}]({source.url})"
            )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    export_payload = {
        "app": APP_TITLE,
        "version": APP_VERSION,
        "research_inputs": research_inputs,
        "sources": [
            source.model_dump()
            for source in sources
        ],
        "original_recipes": [
            recipe.model_dump()
            for recipe in original_recipes
        ],
        "adapted_recipes": [
            recipe.model_dump()
            for recipe in final_recipes
        ],
    }

    export_json = json.dumps(
        export_payload,
        ensure_ascii=False,
        indent=2
    )

    st.download_button(
        "⬇️ EXPORT R&D JSON",
        data=export_json,
        file_name="tasyte_recipe_rd.json",
        mime="application/json",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Tasyte Global Recipe R&D • "
    "Automatic web research + Indian culinary adaptation"
)
