"""
🌿 PlantDoc AI — Hugging Face Spaces
LangChain + LangGraph + BLIP-2 + Groq + Tavily
✅ Optimized: GPU support, diagnosis cache, parallel search, faster inference
"""

import os
import re
import uuid
import torch
import joblib
import asyncio
import hashlib
import pandas as pd
import fitz  # PyMuPDF
import gradio as gr
from PIL import Image
from typing import Optional, Union
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

# ── LangChain / LangGraph ─────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient

# ── Transformers / PEFT ───────────────────────────────────────
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
)

# ── Performance ───────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ══════════════════════════════════════════════════════════════
# ENV / SECRETS
# ══════════════════════════════════════════════════════════════
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
TAVILY_API_KEY    = os.environ.get("TAVILY_API_KEY", "")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")

if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"]    = "PlantDoc-AI"
    os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"

# ══════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════
# BLIP-2
# ══════════════════════════════════════════════════════════════
BLIP2_MODEL_ID = os.environ.get(
    "BLIP2_MODEL_ID",
    "Mahmoud-Badr-Zidan/blip2-plant-disease"
)

print("⏳ Loading BLIP-2 Processor...")
infer_processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)
print("✅ Processor loaded")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
) if DEVICE == "cuda" else None

print("⏳ Loading BLIP-2 Model...")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    BLIP2_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto" if DEVICE == "cuda" else None,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
if DEVICE == "cpu":
    blip_model = blip_model.to("cpu")
blip_model.eval()
print("✅ BLIP-2 Model loaded")

# ══════════════════════════════════════════════════════════════
# DIAGNOSIS CACHE — avoids re-running BLIP-2 for same image
# ══════════════════════════════════════════════════════════════
_diagnosis_cache: dict = {}

def _image_hash(image_path: str) -> str:
    """Fast hash from file path + mtime — no need to read pixels."""
    try:
        mtime = os.path.getmtime(image_path)
        return hashlib.md5(f"{image_path}:{mtime}".encode()).hexdigest()
    except Exception:
        return image_path

# ══════════════════════════════════════════════════════════════
# IRRIGATION MODEL
# ══════════════════════════════════════════════════════════════
IRRIGATION_MODEL_PATH      = os.environ.get("IRRIGATION_MODEL_PATH", "smart_irrigation_model.pkl")
IRRIGATION_PREPROCESS_PATH = os.environ.get("IRRIGATION_PREPROCESS_PATH", "smart_irrigation_preprocessor.pkl")

irrigation_model = irrigation_preprocess = None
try:
    if os.path.exists(IRRIGATION_MODEL_PATH) and os.path.exists(IRRIGATION_PREPROCESS_PATH):
        irrigation_model      = joblib.load(IRRIGATION_MODEL_PATH)
        irrigation_preprocess = joblib.load(IRRIGATION_PREPROCESS_PATH)
        print("✅ Irrigation model loaded")
    else:
        print("⚠️  Irrigation model files not found -- irrigation tool will be disabled")
except Exception as e:
    print(f"⚠️  Irrigation model failed to load: {e} -- continuing without it")

# ══════════════════════════════════════════════════════════════
# PDF reference book
# ══════════════════════════════════════════════════════════════
PDF_PATH = os.environ.get("PDF_PATH", "reference.pdf")
PDF_TEXT = ""
PDF_NAME = ""

if os.path.exists(PDF_PATH):
    doc = fitz.open(PDF_PATH)
    pages_text = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text").strip()
        if text:
            pages_text.append(f"--- Page {page_num + 1} ---\n{text}")
    doc.close()
    PDF_TEXT = "\n\n".join(pages_text)
    PDF_NAME = os.path.basename(PDF_PATH)
    print(f"✅ PDF loaded: {PDF_NAME} ({len(PDF_TEXT):,} chars)")
else:
    print("⚠️  Reference PDF not found — search_in_references will use web only")

# Pre-split PDF pages once for faster searching
PDF_PAGES = PDF_TEXT.split("--- Page ") if PDF_TEXT else []

# ══════════════════════════════════════════════════════════════
# KNOWN PLANTS / DISEASES
# ══════════════════════════════════════════════════════════════
KNOWN_PLANTS = [
    "apple","grape","corn","tomato","potato","strawberry","peach",
    "cherry","pepper","squash","raspberry","soybean","wheat",
    "rice","cucumber","maize","blueberry","orange",
]

KNOWN_DISEASES = [
    "black rot","early blight","late blight","leaf scorch","rust",
    "common rust","northern leaf blight","gray leaf spot","scab",
    "powdery mildew","downy mildew","anthracnose","mosaic",
    "bacterial spot","septoria","target spot","leaf mold",
    "fire blight","crown gall","cercospora","alternaria",
    "botrytis","fusarium","verticillium","phytophthora",
]


def predict_plant_and_disease(image: Image.Image, cache_key: str = "") -> dict:
    """Run BLIP-2 inference. Returns cached result if same image."""
    if cache_key and cache_key in _diagnosis_cache:
        print(f"📦 Cache hit for {cache_key[:8]}")
        return _diagnosis_cache[cache_key]

    prompt = (
        "Question: Identify the plant and the disease. "
        "Answer in this format - Plant: [name], Disease: [name].\nAnswer:"
    )
    inputs = infer_processor(images=image, text=prompt, return_tensors="pt")
    device = next(blip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = blip_model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=1,
            do_sample=False,
            repetition_penalty=1.2,
        )

    full_text   = infer_processor.decode(out[0], skip_special_tokens=True)
    result_text = full_text.split("Answer:")[-1].strip()

    plant_match   = re.search(r"Plant:\s*([^,.\n/\[\]]+)", result_text, re.I)
    disease_match = re.search(r"Disease:\s*([^,.\n/\[\]]+)", result_text, re.I)
    plant   = plant_match.group(1).strip()   if plant_match   else ""
    disease = disease_match.group(1).strip() if disease_match else ""

    if not plant or plant.startswith("["):    plant = ""
    if not disease or disease.startswith("["): disease = ""

    text_low = result_text.lower()
    if not plant:   plant   = next((p.title() for p in KNOWN_PLANTS   if p in text_low), "")
    if not disease:
        if "healthy" in text_low: disease = "healthy"
        else: disease = next((d.title() for d in KNOWN_DISEASES if d in text_low), "")

    plant   = plant   if plant   else "Unknown"
    disease = disease if disease else "Unknown/Healthy"

    result = {"plant": plant, "disease": disease}

    if cache_key:
        _diagnosis_cache[cache_key] = result

    return result


# ══════════════════════════════════════════════════════════════
# DISEASE SYNONYMS
# ══════════════════════════════════════════════════════════════
DISEASE_SYNONYMS = {
    "common rust":          ["corn rust","puccinia sorghi","rust of corn","rust of maize","maize rust"],
    "northern leaf blight": ["exserohilum turcicum","helminthosporium turcicum","turcicum blight","nlb"],
    "gray leaf spot":       ["cercospora zeae-maydis","cercospora leaf spot of corn"],
    "early blight":         ["alternaria blight","alternaria solani","target spot tomato"],
    "late blight":          ["phytophthora blight","phytophthora infestans"],
    "septoria leaf spot":   ["septoria lycopersici","septoria blight"],
    "target spot":          ["corynespora cassiicola"],
    "leaf mold":            ["fulvia fulva","cladosporium fulvum"],
    "bacterial spot":       ["xanthomonas","bacterial leaf spot"],
    "mosaic":               ["tobacco mosaic virus","tomato mosaic virus","tmv","tomv"],
    "black rot":            ["botryosphaeria obtusa","physalospora obtusa","black rot of apple"],
    "scab":                 ["venturia inaequalis","apple scab"],
    "fire blight":          ["erwinia amylovora"],
    "cedar apple rust":     ["gymnosporangium juniperi-virginianae","apple rust"],
    "esca":                 ["black measles","esca disease","phaeomoniella chlamydospora",
                             "phaeoacremonium","esca complex","grapevine trunk disease",
                             "measles of grape","apoplexy","grape esca"],
    "powdery mildew":       ["uncinula necator","erysiphe necator","erysiphe","sphaerotheca"],
    "downy mildew":         ["plasmopara viticola","plasmopara","peronospora"],
    "leaf scorch":          ["cercospora leaf spot","angular leaf spot"],
    "anthracnose":          ["colletotrichum","glomerella"],
    "crown gall":           ["agrobacterium tumefaciens"],
}


def _get_search_terms(plant: str, disease: str) -> list:
    disease_low = disease.lower().strip()
    plant_low   = plant.lower().strip()
    terms = [disease_low]
    if plant_low: terms.append(f"{plant_low} {disease_low}")
    for key, aliases in DISEASE_SYNONYMS.items():
        if disease_low == key or disease_low in aliases:
            terms += aliases[:3]
            break
    return list(dict.fromkeys(terms))


# ══════════════════════════════════════════════════════════════
# PARALLEL SEARCH EXECUTOR
# ══════════════════════════════════════════════════════════════
_search_executor = ThreadPoolExecutor(max_workers=4)


# ══════════════════════════════════════════════════════════════
# LangChain TOOLS
# ══════════════════════════════════════════════════════════════
class LeafImageInput(BaseModel):
    image_path: str = Field(description="Absolute path to the plant leaf image file on disk")

@tool(args_schema=LeafImageInput)
def analyze_leaf_image(image_path: str) -> str:
    """Analyze a plant leaf image using BLIP-2 fine-tuned model.
    Returns plant name and disease name.
    ALWAYS call this FIRST when given an image path."""
    try:
        cache_key = _image_hash(image_path)
        image     = Image.open(image_path).convert("RGB")
        result    = predict_plant_and_disease(image, cache_key=cache_key)

        plant   = result.get("plant",   "Unknown").strip()
        disease = result.get("disease", "Unknown").strip()

        if plant.startswith("[") and plant.endswith("]"):     plant = "Unknown"
        if disease.startswith("[") and disease.endswith("]"): disease = "Unknown"

        HEALTHY_KW = {"healthy","health","normal","no disease","unknown/healthy","none"}
        if any(kw in disease.lower() for kw in HEALTHY_KW) or disease.lower() in {"unknown","none",""}:
            return f"Plant: {plant} | Disease: healthy | STATUS: NO_DISEASE_DETECTED"
        return f"Plant: {plant} | Disease: {disease}"
    except FileNotFoundError:
        return f"Error: Image not found at '{image_path}'"
    except Exception as e:
        return f"BLIP-2 error: {e}"


class ReferenceInput(BaseModel):
    plant:   str = Field(description="Plant name")
    disease: str = Field(description="Disease name")

@tool(args_schema=ReferenceInput)
def search_in_references(plant: str, disease: str) -> str:
    """Search the scientific PDF reference book for information about a plant disease."""
    if not PDF_PAGES:
        return "Reference PDF not available."

    search_terms = _get_search_terms(plant, disease)
    results = []

    for term in search_terms:
        for page_block in PDF_PAGES:
            if term in page_block.lower():
                snippet = page_block[:800].strip()
                results.append(f"[Ref match: '{term}']\n{snippet}")
                if len(results) >= 3:
                    break
        if len(results) >= 3:
            break

    return "\n\n---\n\n".join(results) if results else f"No reference found for '{disease}' in the book."


class TavilyInput(BaseModel):
    query: str = Field(description="Search query for plant disease information")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

@tool(args_schema=TavilyInput)
def search_web_for_disease(query: str) -> str:
    """Search the web for plant disease treatment, prevention, or general information."""
    if not tavily_client:
        return "Web search not available (TAVILY_API_KEY not set)."
    try:
        result   = tavily_client.search(query, max_results=4, search_depth="advanced")
        snippets = [r.get("content","") for r in result.get("results", [])]
        return "\n\n---\n\n".join(snippets[:4]) if snippets else "No results found."
    except Exception as e:
        return f"Web search error: {e}"


# ✅ Combined parallel search tool
class CombinedSearchInput(BaseModel):
    plant:   str = Field(description="Plant name from diagnosis")
    disease: str = Field(description="Disease name from diagnosis")

@tool(args_schema=CombinedSearchInput)
def search_all_sources(plant: str, disease: str) -> str:
    """Search BOTH the PDF reference book AND the web simultaneously (parallel).
    Use this instead of calling search_in_references and search_web_for_disease separately.
    Much faster — both searches run at the same time."""
    query = f"{plant} {disease} treatment prevention symptoms"

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_ref = pool.submit(_pdf_search_raw, plant, disease)
        f_web = pool.submit(_web_search_raw, query)
        ref_result = f_ref.result(timeout=30)
        web_result = f_web.result(timeout=30)

    output = ""
    if ref_result and "No reference found" not in ref_result:
        output += f"📚 **From Reference Book:**\n{ref_result}\n\n"
    if web_result and "No results" not in web_result:
        output += f"🌐 **From Web:**\n{web_result}"

    return output if output else f"No information found for {plant} - {disease}."


def _pdf_search_raw(plant: str, disease: str) -> str:
    """Internal: PDF search without tool wrapper."""
    if not PDF_PAGES:
        return "Reference PDF not available."
    search_terms = _get_search_terms(plant, disease)
    results = []
    for term in search_terms:
        for page_block in PDF_PAGES:
            if term in page_block.lower():
                results.append(f"[Ref match: '{term}']\n{page_block[:800].strip()}")
                if len(results) >= 3:
                    break
        if len(results) >= 3:
            break
    return "\n\n---\n\n".join(results) if results else f"No reference found for '{disease}'."


def _web_search_raw(query: str) -> str:
    """Internal: web search without tool wrapper."""
    if not tavily_client:
        return "Web search not available."
    try:
        result   = tavily_client.search(query, max_results=4, search_depth="advanced")
        snippets = [r.get("content","") for r in result.get("results", [])]
        return "\n\n---\n\n".join(snippets[:4]) if snippets else "No results found."
    except Exception as e:
        return f"Web search error: {e}"


# ══════════════════════════════════════════════════════════════
# IRRIGATION TOOL — ✅ FIXED: accepts Union[float, str, None]
# and coerces to float internally so LLM string values don't crash
# ══════════════════════════════════════════════════════════════
class IrrigationInput(BaseModel):
    plant:         str                        = Field(description="Plant name")
    disease:       str                        = Field(description="Detected disease or 'healthy'")
    temperature_c: Optional[Union[float, str]] = Field(None, description="Air temperature in °C")
    humidity:      Optional[Union[float, str]] = Field(None, description="Relative humidity %")
    soil_moisture: Optional[Union[float, str]] = Field(None, description="Soil moisture %")

@tool(args_schema=IrrigationInput)
def get_irrigation_recommendation(
    plant: str,
    disease: str,
    temperature_c: Optional[Union[float, str]] = None,
    humidity:      Optional[Union[float, str]] = None,
    soil_moisture: Optional[Union[float, str]] = None,
) -> str:
    """Predict irrigation need using the smart irrigation ML model."""
    if irrigation_model is None:
        return "Irrigation model not loaded. Please ensure model .pkl files are present."

    # ✅ FIX: coerce any string/None values coming from the LLM to float
    def _to_float(v, default: float) -> float:
        try:
            return float(v) if v is not None else default
        except (ValueError, TypeError):
            return default

    temperature_c = _to_float(temperature_c, 25.0)
    humidity      = _to_float(humidity,      60.0)
    soil_moisture = _to_float(soil_moisture, 50.0)

    try:
        features = {
            "Plant":         plant,
            "Disease":       disease,
            "Temperature_C": temperature_c,
            "Humidity":      humidity,
            "Soil_Moisture": soil_moisture,
        }
        df   = pd.DataFrame([features])
        X    = irrigation_preprocess.transform(df)
        pred = irrigation_model.predict(X)[0]
        proba = irrigation_model.predict_proba(X)[0]
        conf  = max(proba) * 100
        label = "💧 Irrigation Needed" if pred == 1 else "✅ No Irrigation Needed"
        return (
            f"{label}\n"
            f"Confidence: {conf:.1f}%\n"
            f"Conditions — Temp: {temperature_c}°C | "
            f"Humidity: {humidity}% | "
            f"Soil Moisture: {soil_moisture}%"
        )
    except Exception as e:
        return f"Irrigation prediction error: {e}"


# ══════════════════════════════════════════════════════════════
# LLM + AGENT
# ══════════════════════════════════════════════════════════════
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

llm = ChatOpenAI(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    temperature=0.1,
    max_tokens=1000,
)

SYSTEM_PROMPT = """You are PlantDoc AI — an expert agricultural assistant.

Tools available:
1. analyze_leaf_image   — ALWAYS call first when given an image path
2. search_all_sources   — searches PDF reference book AND web simultaneously (FASTER than calling them separately)
3. search_in_references — use only if you need PDF-only results
4. search_web_for_disease — use only if you need web-only results
5. get_irrigation_recommendation — predict irrigation need

Workflow for image diagnosis (optimized):
1. Call analyze_leaf_image immediately
2. Call search_all_sources with detected plant + disease (replaces two separate calls)
3. Call get_irrigation_recommendation ONLY if environmental conditions are provided
4. Return structured report: diagnosis, symptoms, treatment, prevention, irrigation

Workflow for text questions (no image):
1. Call search_all_sources directly with the plant and disease mentioned
2. Return a focused answer

Always respond in the language the user uses (Arabic or English).
Be concise — users are waiting for a fast response.
"""

memory = MemorySaver()
tools  = [
    analyze_leaf_image,
    search_all_sources,
    search_in_references,
    search_web_for_disease,
    get_irrigation_recommendation,
]
agent = create_react_agent(llm, tools, checkpointer=memory, prompt=SYSTEM_PROMPT)
print("✅ LangGraph agent ready!")

# ══════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════════
SESSION_THREADS: dict = {}

def get_or_create_thread(session_id: str) -> str:
    if session_id not in SESSION_THREADS:
        SESSION_THREADS[session_id] = f"plantdoc-{session_id[:8]}"
    return SESSION_THREADS[session_id]

# ══════════════════════════════════════════════════════════════
# INTENT DETECTION
# ══════════════════════════════════════════════════════════════
INTENT_MAP = {
    "DIAGNOSIS":   ["what disease","identify","healthy or diseased","is the plant","تشخيص","مرض"],
    "SYMPTOMS":    ["symptom","signs","look like","visual","أعراض","علامات"],
    "SEVERITY":    ["how severe","severity","early stage","late stage","شدة","خطورة"],
    "CAUSES":      ["what cause","humidity","overwatering","pathogen","سبب","رطوبة"],
    "TREATMENT":   ["treat","treatment","fungicide","organic","cure","علاج","مبيد"],
    "PREVENTION":  ["prevent","prevention","avoid","resistant","وقاية","منع"],
    "IRRIGATION":  ["irrigation","water","watering","soil moisture","ري","سقي","مياه"],
    "DESCRIPTION": ["describe","life cycle","scientific name","biology","وصف","دورة حياة"],
}

INTENT_INSTRUCTION = {
    "DIAGNOSIS":   "FOCUSED: Reply with ONLY 🌿 Plant, 🦠 Disease, ⚠️ Type.",
    "SYMPTOMS":    "FOCUSED: Reply with ONLY 🔍 Visual Signs bullet list.",
    "SEVERITY":    "FOCUSED: Reply with ONLY 📊 Severity level and 1-2 sentences.",
    "CAUSES":      "FOCUSED: Reply with ONLY 🌧️ Cause/Environment section.",
    "TREATMENT":   "FOCUSED: Reply with ONLY 💊 Treatment numbered steps.",
    "PREVENTION":  "FOCUSED: Reply with ONLY 🛡️ Prevention numbered steps.",
    "IRRIGATION":  "FOCUSED: Reply with ONLY 💧 Irrigation Need and 🌡️ Conditions.",
    "DESCRIPTION": "FOCUSED: Reply with ONLY 📋 Description paragraph.",
}

def detect_intent(text: str):
    text_lower = text.lower()
    for intent, keywords in INTENT_MAP.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return None


# ══════════════════════════════════════════════════════════════
# CHAT FUNCTION — with streaming support
# ══════════════════════════════════════════════════════════════
def chat_with_plantdoc(message, image, temperature_c, humidity, soil_moisture, history, session_id):
    thread_id = get_or_create_thread(session_id)

    weather_data = {}
    if temperature_c: weather_data["temperature_c"] = temperature_c
    if humidity:      weather_data["humidity"]      = humidity
    if soil_moisture: weather_data["soil_moisture"] = soil_moisture

    if image is not None:
        image_path = image
        user_text  = f"Diagnose the plant disease in this image: {image_path}"
        if message.strip():
            user_text += f"\nAdditional context: {message.strip()}"
        intent = detect_intent(message) if message.strip() else None
    else:
        if not message.strip():
            yield history + [{"role": "assistant", "content": "⚠️ Please type a message or upload a plant image."}], session_id
            return
        user_text  = message.strip()
        image_path = ""
        intent     = detect_intent(message)

    focus_hint = f"\n\n{INTENT_INSTRUCTION[intent]}" if intent and intent in INTENT_INSTRUCTION else ""

    display_msg = message.strip() if message.strip() else "📸 (image uploaded for diagnosis)"
    history = history + [{"role": "user", "content": display_msg}]
    yield history, session_id

    weather_str = "\n".join([f"  - {k}: {v}" for k, v in weather_data.items()]) if weather_data else "  - Not provided"

    if image_path:
        full_message = (
            f"{user_text}\n\n"
            f"IMAGE_PATH: {image_path}\n"
            f"Weather/Soil conditions:\n{weather_str}"
            f"{focus_hint}\n\n"
            f'ACTION: Call analyze_leaf_image(image_path="{image_path}") immediately as the first step.'
        )
    else:
        full_message = user_text + focus_hint

    config = {"configurable": {"thread_id": thread_id}}

    try:
        partial_text     = ""
        tool_calls_count = 0
        thinking_shown   = False

        for chunk in agent.stream(
            {"messages": [HumanMessage(content=full_message)]},
            config=config,
            stream_mode="updates",
        ):
            if not thinking_shown:
                history_stream = history + [{"role": "assistant", "content": "🔍 جاري التحليل…"}]
                yield history_stream, session_id
                thinking_shown = True

            for node_output in chunk.values():
                msgs = node_output.get("messages", [])
                for m in msgs:
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        tool_calls_count += len(m.tool_calls)
                        for tc in m.tool_calls:
                            tool_name = tc.get("name", "")
                            status_icons = {
                                "analyze_leaf_image":           "🔬 جاري تحليل الصورة…",
                                "search_all_sources":           "🔍 جاري البحث في المراجع والويب…",
                                "search_in_references":         "📚 جاري البحث في المرجع…",
                                "search_web_for_disease":       "🌐 جاري البحث على الويب…",
                                "get_irrigation_recommendation":"💧 جاري حساب توصيات الري…",
                            }
                            status = status_icons.get(tool_name, f"⚙️ {tool_name}…")
                            history_stream = history + [{"role": "assistant", "content": status}]
                            yield history_stream, session_id

                    if isinstance(m, AIMessage) and hasattr(m, "content") and m.content:
                        if isinstance(m.content, str) and m.content.strip():
                            partial_text = m.content

        if not partial_text:
            partial_text = "⚠️ لم أتمكن من إنتاج رد. حاول مرة أخرى."

        if tool_calls_count:
            partial_text += f"\n\n---\n🔧 *{tool_calls_count} tool call(s) | Thread: `{thread_id}`*"

    except Exception as e:
        partial_text = f"❌ Agent error: {e}"

    history = history + [{"role": "assistant", "content": partial_text}]
    yield history, session_id


# ══════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════
QUESTION_CATEGORIES = {
    "🔬 تشخيص": [
        "What plant disease is visible in this image?",
        "Identify the disease affecting this leaf.",
        "Is the plant healthy or diseased?",
        "What symptoms indicate the detected disease?",
        "How severe is the infection?",
    ],
    "🌧️ الأسباب": [
        "What environmental conditions may have caused this disease?",
        "Is high humidity contributing to the disease?",
        "Could overwatering be causing the symptoms?",
        "What pathogens are responsible for this disease?",
        "How does temperature affect the spread of this disease?",
    ],
    "💊 العلاج": [
        "What is the recommended treatment for this disease?",
        "Which fungicides are effective against this disease?",
        "Are there organic treatments available?",
        "How quickly should I apply treatment?",
        "What dosage of fungicide should I use?",
    ],
    "🛡️ الوقاية": [
        "How can I prevent this disease from spreading?",
        "What resistant plant varieties should I use?",
        "How should I adjust irrigation to prevent this disease?",
        "Should I remove infected leaves immediately?",
        "What crop rotation strategy helps prevent this disease?",
    ],
    "💧 الري": [
        "What is the recommended irrigation level for this plant?",
        "How does soil moisture affect this disease?",
        "Should I reduce watering frequency?",
        "What soil type is best to prevent this disease?",
        "How does drainage affect disease development?",
    ],
    "📋 معلومات": [
        "Describe the disease biology and life cycle.",
        "What is the scientific name of the pathogen?",
        "How does this disease spread between plants?",
        "What crops are most susceptible to this disease?",
        "What is the economic impact of this disease?",
    ],
}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg: #0a0f0a;
    --surface: #111811;
    --surface2: #172017;
    --border: #1e2e1e;
    --green-bright: #4ade80;
    --green-mid: #22c55e;
    --green-dark: #166534;
    --green-glow: rgba(74,222,128,0.15);
    --text: #e2f5e2;
    --text-muted: #6b8f6b;
    --accent: #86efac;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Tajawal', sans-serif !important;
}

.plantdoc-header {
    text-align: center;
    padding: 32px 20px 16px;
    position: relative;
}
.plantdoc-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: var(--green-bright);
    letter-spacing: -1px;
    margin: 0;
    text-shadow: 0 0 40px rgba(74,222,128,0.4);
}
.plantdoc-header p {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin: 8px 0 0;
}

.gr-panel, .gr-box, .gr-form, .gr-block {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

.gr-chatbot {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.message.user {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--accent) !important;
    border-radius: 10px 10px 2px 10px !important;
}
.message.bot {
    background: linear-gradient(135deg, #0f1f0f, #111811) !important;
    border: 1px solid #1e3a1e !important;
    color: var(--text) !important;
    border-radius: 10px 10px 10px 2px !important;
}

.gr-image {
    border: 2px dashed var(--green-dark) !important;
    border-radius: 12px !important;
    background: var(--surface2) !important;
    transition: border-color 0.3s;
}
.gr-image:hover {
    border-color: var(--green-bright) !important;
}

input[type=range] {
    accent-color: var(--green-mid) !important;
}
.gr-slider-container label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}

textarea, input[type=text] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'Tajawal', sans-serif !important;
}
textarea:focus, input[type=text]:focus {
    border-color: var(--green-mid) !important;
    box-shadow: 0 0 0 3px var(--green-glow) !important;
}

button.primary {
    background: linear-gradient(135deg, var(--green-mid), var(--green-dark)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    background: linear-gradient(135deg, var(--green-bright), var(--green-mid)) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px var(--green-glow) !important;
}
button.secondary {
    background: var(--surface2) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
button.secondary:hover {
    border-color: var(--green-mid) !important;
    color: var(--green-bright) !important;
}

.cat-btn button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    padding: 6px 14px !important;
    transition: all 0.2s !important;
}
.cat-btn button:hover {
    background: var(--green-glow) !important;
    border-color: var(--green-mid) !important;
    color: var(--green-bright) !important;
}

.q-btn button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-size: 0.75rem !important;
    text-align: right !important;
    white-space: normal !important;
    height: auto !important;
    padding: 8px 12px !important;
    transition: all 0.2s !important;
    line-height: 1.4 !important;
}
.q-btn button:hover {
    border-color: var(--green-mid) !important;
    color: var(--green-bright) !important;
    background: var(--green-glow) !important;
}

label, .gr-label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}

hr { border-color: var(--border) !important; }

.section-title {
    color: var(--green-bright);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 16px 0 8px;
    opacity: 0.7;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--green-dark); border-radius: 2px; }
"""

HEADER_HTML = """
<div class="plantdoc-header">
    <h1>🌿 PlantDoc AI</h1>
    <p>ارفع صورة ورقة نبات واحصل على تشخيص فوري للمرض والعلاج</p>
</div>
"""

def new_session():
    return str(uuid.uuid4()), []

with gr.Blocks(css=CUSTOM_CSS, title="PlantDoc AI", theme=gr.themes.Base()) as demo:
    session_id = gr.State(value=str(uuid.uuid4()))

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=260):
            image_input = gr.Image(
                type="filepath",
                label="📸 صورة الورقة",
                height=200,
            )
            gr.HTML('<div class="section-title">🌡 الظروف البيئية (اختياري)</div>')
            # ✅ FIX: default values set to sensible numbers instead of None
            # to prevent Gradio from sending the string "None" to the agent
            temp_slider  = gr.Slider(0, 50,  step=0.5, label="درجة الحرارة (°C)", value=25.0)
            hum_slider   = gr.Slider(0, 100, step=1,   label="الرطوبة (%)",       value=60.0)
            moist_slider = gr.Slider(0, 100, step=1,   label="رطوبة التربة (%)",  value=50.0)
            new_session_btn = gr.Button("↺  جلسة جديدة", variant="secondary", size="sm")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="",
                type="messages",
                height=420,
                avatar_images=(
                    None,
                    "https://em-content.zobj.net/source/apple/391/potted-plant_1fab4.png"
                ),
                show_copy_button=True,
                placeholder="<div style='text-align:center;color:#4ade80;padding:40px;font-family:Space Mono,monospace;'>🌿<br><br>ارفع صورة أو اكتب سؤالك<br><span style='font-size:0.7rem;opacity:0.5'>PlantDoc AI جاهز</span></div>",
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="اكتب سؤالك هنا…",
                    label="",
                    scale=5,
                    lines=1,
                    max_lines=3,
                )
                send_btn = gr.Button("إرسال ➤", variant="primary", scale=1, min_width=80)

    gr.HTML('<hr><div class="section-title" style="text-align:center">💬 أسئلة سريعة</div>')

    cat_names = list(QUESTION_CATEGORIES.keys())

    with gr.Row(elem_classes=["cat-btn"]):
        cat_btns = [
            gr.Button(cat, size="sm", variant="secondary")
            for cat in cat_names
        ]

    cat_question_groups = {}
    for cat, questions in QUESTION_CATEGORIES.items():
        visible = (cat == cat_names[0])
        with gr.Group(visible=visible) as grp:
            rows   = [questions[i:i+2] for i in range(0, len(questions), 2)]
            q_btns = []
            for row_qs in rows:
                with gr.Row(elem_classes=["q-btn"]):
                    for q in row_qs:
                        b = gr.Button(q, size="sm")
                        q_btns.append(b)
        cat_question_groups[cat] = (grp, q_btns)

    # ── Events ──────────────────────────────────────────────────
    send_inputs  = [msg_box, image_input, temp_slider, hum_slider, moist_slider, chatbot, session_id]
    send_outputs = [chatbot, session_id]

    send_btn.click(fn=chat_with_plantdoc, inputs=send_inputs, outputs=send_outputs).then(fn=lambda: "", outputs=msg_box)
    msg_box.submit(fn=chat_with_plantdoc, inputs=send_inputs, outputs=send_outputs).then(fn=lambda: "", outputs=msg_box)
    new_session_btn.click(fn=new_session, outputs=[session_id, chatbot])

    all_groups = [cat_question_groups[c][0] for c in cat_names]
    for i, (cat_btn, cat_name) in enumerate(zip(cat_btns, cat_names)):
        def make_visibility(selected_idx):
            def fn():
                return [gr.update(visible=(j == selected_idx)) for j in range(len(cat_names))]
            return fn
        cat_btn.click(fn=make_visibility(i), outputs=all_groups)

    quick_inputs  = [image_input, temp_slider, hum_slider, moist_slider, chatbot, session_id]
    quick_outputs = [msg_box, chatbot, session_id]

    def make_sender(q):
        def fn(image, temp, hum, moist, history, sid):
            results = list(chat_with_plantdoc(q, image, temp, hum, moist, history, sid))
            last = results[-1]
            return ("",) + last
        return fn

    for cat, (grp, q_btns) in cat_question_groups.items():
        for btn in q_btns:
            q_text = btn.value
            btn.click(fn=make_sender(q_text), inputs=quick_inputs, outputs=quick_outputs)

demo.launch()
