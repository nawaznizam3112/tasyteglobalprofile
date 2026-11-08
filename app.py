import os
import re
import json
import hashlib
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
import streamlit as st

from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="MasalaAtlas AI",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "MasalaAtlas AI"

# You can change this to another available Gemini model.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
    .main {
        background: #faf8f3;
    }

    .hero {
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        border-radius: 18px;
        background:
            linear-gradient(
                135deg,
                rgba(255, 153, 51, 0.16),
                rgba(210, 45, 45, 0.08)
            );
        border: 1px solid rgba(150, 80, 20, 0.15);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin-bottom: 0.2rem;
    }

    .recipe-card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #e5ded1;
        background: white;
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        background: white;
        border: 1px solid #e5ded1;
        text-align: center;
    }

    .small {
        color: #6b6258;
        font-size: 0.88rem;
    }

    .tag {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        margin-right: 0.3rem;
        border-radius: 999px;
        background: #f2eadf;
        font-size: 0.78rem;
    }

    .warning {
        padding: 0.8rem;
        background: #fff4d6;
        border-radius: 10px;
        border-left: 4px solid #d99a00;
    }

    .success {
        padding: 0.8rem;
        background: #eaf7ed;
        border-radius: 10px;
        border-left: 4px solid #31914b;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class Ingredient(BaseModel):
    name: str
    quantity: str
    notes: Optional[str] = ""


class Recipe(BaseModel):
    title: str
    source_url: Optional[str] = ""
    source_country: Optional[str] = ""
    source_region: Optional[str] = ""

    description: str = ""

    cuisine: str = ""
    dish_type: str = ""

    ingredients: List[Ingredient] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)

    prep_time_minutes: int = 0
    cook_time_minutes: int = 0
    servings: int = 1

    dietary_tags: List[str] = Field(default_factory=list)
    allergens: List[str] = Field(default_factory=list)


class AdaptedRecipe(Recipe):
    indian_name: str = ""

    target_indian_region: str = ""
    flavour_profile: str = ""

    spice_level: int = 5
    tanginess: int = 5
    sweetness: int = 3
    saltiness: int = 5
    richness: int = 5
    smokiness: int = 3

    street_food_score: int = 0
    commercial_score: int = 0

    indian_adaptation_notes: List[str] = Field(default_factory=list)
    substitutions: List[str] = Field(default_factory=list)

    plating_and_serving: List[str] = Field(default_factory=list)
    batch_production_notes: List[str] = Field(default_factory=list)

    estimated_food_cost_inr: Optional[float] = None
    suggested_selling_price_inr: Optional[float] = None


# ============================================================
# GEMINI CLIENT
# ============================================================

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
        return None

    return genai.Client(api_key=api_key)


def gemini_available():
    return get_gemini_client() is not None


# ============================================================
# UTILITIES
# ============================================================

def clean_text(text: str, max_chars: int = 18000) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_chars]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_json(text: str):
    """
    Defensive JSON extraction in case a model returns markdown fences.
    """
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not extract JSON from Gemini response.")


# ============================================================
# WEB SCRAPER
# ============================================================

def scrape_recipe_url(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/125.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    # Remove noisy elements.
    for element in soup(
        ["script", "style", "noscript", "svg", "nav", "footer"]
    ):
        element.decompose()

    # First preference: JSON-LD recipe schema.
    jsonld_recipes = []

    for script in soup.find_all(
        "script",
        type="application/ld+json",
    ):
        try:
            data = json.loads(script.string or script.get_text())
        except Exception:
            continue

        if isinstance(data, dict):
            if data.get("@type") == "Recipe":
                jsonld_recipes.append(data)

            elif "@graph" in data:
                for item in data["@graph"]:
                    if (
                        isinstance(item, dict)
                        and item.get("@type") == "Recipe"
                    ):
                        jsonld_recipes.append(item)

        elif isinstance(data, list):
            for item in data:
                if (
                    isinstance(item, dict)
                    and item.get("@type") == "Recipe"
                ):
                    jsonld_recipes.append(item)

    text = clean_text(soup.get_text(" ", strip=True))

    return {
        "url": url,
        "title": soup.title.get_text(strip=True)
        if soup.title
        else "",
        "jsonld": jsonld_recipes,
        "page_text": text,
    }


# ============================================================
# GEMINI RECIPE EXTRACTION
# ============================================================

def extract_recipe_with_ai(scraped):
    client = get_gemini_client()

    if not client:
        raise RuntimeError(
            "Gemini API key not configured."
        )

    jsonld = json.dumps(
        scraped["jsonld"],
        ensure_ascii=False,
    )

    prompt = f"""
You are a professional culinary data extraction system.

Extract the recipe from the supplied webpage.

IMPORTANT:
- Do not invent missing ingredients.
- Preserve original quantities where possible.
- If a field is unknown, leave it blank.
- Separate ingredients from instructions.
- Identify country/region/cuisine only when reasonably supported.
- This is an extraction task, not an adaptation task.

URL:
{scraped["url"]}

JSON-LD recipe data:
{jsonld[:12000]}

Visible webpage text:
{scraped["page_text"][:16000]}
"""

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Recipe,
            temperature=0.2,
        ),
    )

    if getattr(response, "parsed", None):
        recipe = response.parsed
        if isinstance(recipe, Recipe):
            recipe.source_url = scraped["url"]
            return recipe

    data = extract_json(response.text)

    recipe = Recipe.model_validate(data)
    recipe.source_url = scraped["url"]

    return recipe


# ============================================================
# INDIAN ADAPTATION
# ============================================================

def build_adaptation_prompt(
    recipe: Recipe,
    target_region: str,
    flavour_profile: str,
    spice_level: int,
    tanginess: int,
    sweetness: int,
    saltiness: int,
    richness: int,
    smokiness: int,
    vegetarian: bool,
    vegan: bool,
    jain: bool,
    street_food: bool,
):
    return f"""
You are an expert Indian chef, food R&D consultant and restaurant
menu developer.

Your task is to transform an international recipe into an Indian
street-food-friendly recipe.

DO NOT simply add chilli powder.

Think about:
- Indian ingredient availability
- Indian cooking techniques
- regional flavour traditions
- street-food eating behaviour
- texture
- aroma
- chutneys
- masalas
- frying/grilling/tawa techniques
- batch production
- consistency
- food cost
- speed of service
- holding quality

SOURCE RECIPE:
{recipe.model_dump_json(indent=2)}

TARGET INDIAN REGION:
{target_region}

TARGET FLAVOUR PROFILE:
{flavour_profile}

ADJUSTABLE FLAVOUR LEVELS:
Spice: {spice_level}/10
Tanginess: {tanginess}/10
Sweetness: {sweetness}/10
Saltiness: {saltiness}/10
Richness: {richness}/10
Smokiness: {smokiness}/10

DIETARY REQUIREMENTS:
Vegetarian: {vegetarian}
Vegan: {vegan}
Jain: {jain}

STREET FOOD CONCEPT:
{street_food}

IMPORTANT CULINARY RULES:

1. Preserve the identity of the original dish where useful.
2. Adapt intelligently to Indian taste rather than randomly
   adding Indian ingredients.
3. Use practical Indian ingredients.
4. If an original ingredient is unavailable or unsuitable,
   propose a realistic substitution.
5. Give quantities.
6. Make the recipe executable by a professional kitchen.
7. Suggest sauces/chutneys/accompaniments where appropriate.
8. Consider whether the recipe can be prepared in batches.
9. Consider portion economics.
10. Estimate food cost conservatively in INR.
11. Suggest a commercially sensible selling price.
12. Explain major adaptations.
13. Do not claim exact authenticity unless justified.
14. If the dish would not work well as Indian street food,
    modify the format while preserving its core identity.

The output must be a structured recipe.
"""


def adapt_recipe(
    recipe: Recipe,
    target_region: str,
    flavour_profile: str,
    spice_level: int,
    tanginess: int,
    sweetness: int,
    saltiness: int,
    richness: int,
    smokiness: int,
    vegetarian: bool,
    vegan: bool,
    jain: bool,
    street_food: bool,
):
    client = get_gemini_client()

    if not client:
        raise RuntimeError(
            "Gemini API key not configured."
        )

    prompt = build_adaptation_prompt(
        recipe,
        target_region,
        flavour_profile,
        spice_level,
        tanginess,
        sweetness,
        saltiness,
        richness,
        smokiness,
        vegetarian,
        vegan,
        jain,
        street_food,
    )

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AdaptedRecipe,
            temperature=0.65,
        ),
    )

    if getattr(response, "parsed", None):
        result = response.parsed

        if isinstance(result, AdaptedRecipe):
            return result

    data = extract_json(response.text)

    return AdaptedRecipe.model_validate(data)


# ============================================================
# RECIPE RANKING
# ============================================================

def calculate_business_score(recipe: AdaptedRecipe):
    return round(
        (
            recipe.street_food_score * 0.35
            + recipe.commercial_score * 0.35
            + (10 - recipe.richness) * 0.05
            + recipe.spice_level * 0.05
            + recipe.tanginess * 0.05
            + recipe.smokiness * 0.05
        )
        * 10,
        1,
    )


# ============================================================
# SESSION STATE
# ============================================================

if "recipes" not in st.session_state:
    st.session_state.recipes = []

if "adapted_recipes" not in st.session_state:
    st.session_state.adapted_recipes = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">

# 🌶️ MasalaAtlas AI

### Global recipes → Indian palate → Street-food R&D

Discover recipes from around the world and intelligently transform
them into commercially useful Indian street-food concepts.

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ AI Kitchen Controls")

    mode = st.radio(
        "Mode",
        [
            "User",
            "Admin / R&D",
        ],
    )

    st.divider()

    st.subheader("🇮🇳 Indian Target")

    target_region = st.selectbox(
        "Indian region",
        [
            "Pan-India",
            "Mumbai / Maharashtra",
            "Delhi / NCR",
            "Punjab",
            "Rajasthan",
            "Gujarat",
            "Karnataka",
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
        "Flavour direction",
        [
            "Classic Indian Street Food",
            "Mumbai Street Food",
            "Delhi Street Food",
            "South Indian",
            "Indo-Chinese",
            "Punjabi",
            "Tandoori / Smoky",
            "Chatpata",
            "Spicy & Tangy",
            "Sweet & Spicy",
            "Modern Indian Fusion",
            "Premium Gourmet Street Food",
        ],
    )

    st.subheader("🎚️ Flavour Controls")

    spice_level = st.slider(
        "🌶️ Spice",
        0,
        10,
        6,
    )

    tanginess = st.slider(
        "🍋 Tanginess",
        0,
        10,
        6,
    )

    sweetness = st.slider(
        "🍬 Sweetness",
        0,
        10,
        3,
    )

    saltiness = st.slider(
        "🧂 Saltiness",
        0,
        10,
        5,
    )

    richness = st.slider(
        "🥛 Richness",
        0,
        10,
        5,
    )

    smokiness = st.slider(
        "🔥 Smokiness",
        0,
        10,
        4,
    )

    st.divider()

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
# API STATUS
# ============================================================

if not gemini_available():
    st.warning(
        "Gemini API key is not configured. Add GEMINI_API_KEY "
        "to your environment or Streamlit secrets."
    )


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Recipes collected",
        len(st.session_state.recipes),
    )

with c2:
    st.metric(
        "Indianized",
        len(st.session_state.adapted_recipes),
    )

with c3:
    st.metric(
        "Target region",
        target_region,
    )

with c4:
    st.metric(
        "AI model",
        DEFAULT_MODEL,
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🌍 Discover",
        "🇮🇳 Indianize",
        "📊 R&D Lab",
        "💾 Export",
    ]
)


# ============================================================
# DISCOVER
# ============================================================

with tab1:

    st.header("🌍 Global Recipe Discovery")

    st.info(
        """
        Paste recipe URLs from websites you have permission to scrape.
        The app extracts structured recipe information and then uses
        Gemini only for normalization/interpretation.
        """
    )

    url = st.text_input(
        "Recipe URL",
        placeholder="https://example.com/recipe",
    )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        if st.button(
            "🔎 Scrape & Extract Recipe",
            type="primary",
            use_container_width=True,
        ):

            if not url:
                st.error("Please enter a recipe URL.")

            elif not gemini_available():
                st.error("Configure your Gemini API key first.")

            else:
                try:
                    with st.spinner(
                        "Scraping webpage and extracting recipe..."
                    ):
                        scraped = scrape_recipe_url(url)
                        recipe = extract_recipe_with_ai(scraped)

                    st.session_state.recipes.append(recipe)

                    st.success(
                        f"Recipe extracted: {recipe.title}"
                    )

                except Exception as e:
                    st.error(
                        f"Could not process recipe: {e}"
                    )

    with col_b:
        if st.button(
            "🧹 Clear recipes",
            use_container_width=True,
        ):
            st.session_state.recipes = []
            st.session_state.adapted_recipes = []
            st.rerun()

    st.divider()

    if st.session_state.recipes:

        st.subheader("Collected recipes")

        for i, recipe in enumerate(
            st.session_state.recipes
        ):

            with st.expander(
                f"{i + 1}. {recipe.title}"
            ):

                st.write(
                    recipe.description
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Prep",
                    f"{recipe.prep_time_minutes} min",
                )

                c2.metric(
                    "Cook",
                    f"{recipe.cook_time_minutes} min",
                )

                c3.metric(
                    "Servings",
                    recipe.servings,
                )

                st.markdown("### Ingredients")

                for ingredient in recipe.ingredients:
                    st.write(
                        f"• **{ingredient.quantity}** "
                        f"{ingredient.name} "
                        f"{ingredient.notes or ''}"
                    )

                st.markdown("### Method")

                for step_no, step in enumerate(
                    recipe.instructions,
                    1,
                ):
                    st.write(
                        f"{step_no}. {step}"
                    )


# ============================================================
# INDIANIZE
# ============================================================

with tab2:

    st.header("🇮🇳 Convert to Indian Palate")

    if not st.session_state.recipes:

        st.info(
            "First collect at least one recipe in the Discover tab."
        )

    else:

        recipe_names = [
            r.title
            for r in st.session_state.recipes
        ]

        selected_index = st.selectbox(
            "Select recipe",
            range(len(recipe_names)),
            format_func=lambda x: recipe_names[x],
        )

        selected_recipe = st.session_state.recipes[
            selected_index
        ]

        st.markdown(
            f"""
### Selected recipe

**{selected_recipe.title}**

Source:
`{selected_recipe.source_url or "Unknown"}`
"""
        )

        st.markdown(
            f"""
<div class="recipe-card">

<strong>Target:</strong> {target_region}<br>
<strong>Flavour:</strong> {flavour_profile}<br>
<strong>Spice:</strong> {spice_level}/10<br>
<strong>Tang:</strong> {tanginess}/10<br>
<strong>Sweet:</strong> {sweetness}/10<br>
<strong>Richness:</strong> {richness}/10<br>
<strong>Smoke:</strong> {smokiness}/10

</div>
""",
            unsafe_allow_html=True,
        )

        if st.button(
            "🌶️ Convert recipe",
            type="primary",
            use_container_width=True,
        ):

            if not gemini_available():
                st.error(
                    "Configure Gemini API first."
                )

            else:

                try:

                    with st.spinner(
                        "Chef AI is developing the Indian version..."
                    ):

                        adapted = adapt_recipe(
                            selected_recipe,
                            target_region,
                            flavour_profile,
                            spice_level,
                            tanginess,
                            sweetness,
                            saltiness,
                            richness,
                            smokiness,
                            vegetarian,
                            vegan,
                            jain,
                            street_food,
                        )

                    # Replace previous adaptation for same source.
                    st.session_state.adapted_recipes = [
                        r
                        for r in st.session_state.adapted_recipes
                        if r.source_url
                        != adapted.source_url
                    ]

                    st.session_state.adapted_recipes.append(
                        adapted
                    )

                    st.success(
                        f"Created: {adapted.indian_name}"
                    )

                except Exception as e:
                    st.error(
                        f"Adaptation failed: {e}"
                    )

        if st.session_state.adapted_recipes:

            adapted = st.session_state.adapted_recipes[-1]

            st.divider()

            st.subheader(
                f"🍽️ {adapted.indian_name}"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Street Food Score",
                f"{adapted.street_food_score}/10",
            )

            c2.metric(
                "Commercial Score",
                f"{adapted.commercial_score}/10",
            )

            c3.metric(
                "Estimated Food Cost",
                (
                    f"₹{adapted.estimated_food_cost_inr:.0f}"
                    if adapted.estimated_food_cost_inr
                    else "N/A"
                ),
            )

            c4.metric(
                "Suggested Price",
                (
                    f"₹{adapted.suggested_selling_price_inr:.0f}"
                    if adapted.suggested_selling_price_inr
                    else "N/A"
                ),
            )

            st.markdown(
                f"""
<div class="success">

<strong>Indian positioning</strong><br>

Target region: {adapted.target_indian_region}<br>
Flavour direction: {adapted.flavour_profile}

</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("### 🧂 Ingredients")

            for ingredient in adapted.ingredients:

                notes = (
                    f" — {ingredient.notes}"
                    if ingredient.notes
                    else ""
                )

                st.write(
                    f"• **{ingredient.quantity}** "
                    f"{ingredient.name}{notes}"
                )

            st.markdown("### 👨‍🍳 Preparation")

            for i, step in enumerate(
                adapted.instructions,
                1,
            ):
                st.write(
                    f"{i}. {step}"
                )

            st.markdown(
                "### 🔄 Key Indian adaptations"
            )

            for item in adapted.indian_adaptation_notes:
                st.write(f"• {item}")

            st.markdown(
                "### 🔁 Ingredient substitutions"
            )

            for item in adapted.substitutions:
                st.write(f"• {item}")

            st.markdown(
                "### 🥣 Serving & accompaniments"
            )

            for item in adapted.plating_and_serving:
                st.write(f"• {item}")

            st.markdown(
                "### 🏪 Batch production"
            )

            for item in adapted.batch_production_notes:
                st.write(f"• {item}")


# ============================================================
# R&D LAB
# ============================================================

with tab3:

    st.header("📊 Restaurant R&D Lab")

    if not st.session_state.adapted_recipes:

        st.info(
            "Create Indian adaptations first."
        )

    else:

        data = []

        for recipe in st.session_state.adapted_recipes:

            business_score = calculate_business_score(
                recipe
            )

            data.append(
                {
                    "Recipe": recipe.indian_name,
                    "Region": recipe.target_indian_region,
                    "Street Food": recipe.street_food_score,
                    "Commercial": recipe.commercial_score,
                    "Business Score": business_score,
                    "Cost ₹": recipe.estimated_food_cost_inr,
                    "Price ₹": recipe.suggested_selling_price_inr,
                }
            )

        st.dataframe(
            data,
            use_container_width=True,
        )

        st.divider()

        st.subheader(
            "💡 AI product-development questions"
        )

        question = st.text_area(
            "Ask the AI about your recipe portfolio",
            placeholder=(
                "Which of these recipes is most suitable "
                "for a Bengaluru street-food stall?"
            ),
        )

        if st.button(
            "Ask R&D AI",
            use_container_width=True,
        ):

            if not gemini_available():
                st.error(
                    "Configure Gemini API first."
                )

            elif not question:
                st.warning(
                    "Enter a question."
                )

            else:

                portfolio = json.dumps(
                    [
                        r.model_dump()
                        for r in st.session_state.adapted_recipes
                    ],
                    indent=2,
                    ensure_ascii=False,
                )

                prompt = f"""
You are a restaurant strategy and food R&D consultant.

Here is the current recipe portfolio:

{portfolio[:30000]}

Answer this question:

{question}

Focus on:
- Indian customer appeal
- street-food practicality
- speed of service
- food cost
- ingredient availability
- consistency
- differentiation
- scalability
- menu engineering

Be practical rather than theoretical.
"""

                try:

                    with st.spinner(
                        "Analysing recipe portfolio..."
                    ):

                        client = get_gemini_client()

                        response = client.models.generate_content(
                            model=DEFAULT_MODEL,
                            contents=prompt,
                        )

                    st.markdown(response.text)

                except Exception as e:
                    st.error(str(e))


# ============================================================
# EXPORT
# ============================================================

with tab4:

    st.header("💾 Export Recipe R&D")

    if not st.session_state.adapted_recipes:

        st.info(
            "There are no adapted recipes to export."
        )

    else:

        for i, recipe in enumerate(
            st.session_state.adapted_recipes
        ):

            recipe_json = json.dumps(
                recipe.model_dump(),
                indent=2,
                ensure_ascii=False,
            )

            st.download_button(
                label=f"⬇️ Download {recipe.indian_name}",
                data=recipe_json,
                file_name=(
                    re.sub(
                        r"[^a-zA-Z0-9_-]",
                        "_",
                        recipe.indian_name,
                    )
                    + ".json"
                ),
                mime="application/json",
                key=f"download_{i}",
                use_container_width=True,
            )

            with st.expander(
                f"View JSON — {recipe.indian_name}"
            ):
                st.code(
                    recipe_json,
                    language="json",
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
MasalaAtlas AI — Global recipe intelligence for Indian food R&D.

Use recipe websites in accordance with their terms, robots policies,
copyright/licensing requirements and applicable law. This application
does not bypass authentication, paywalls, anti-bot systems or access
controls.
"""
)
