---
title: PlantDoc AI
emoji: 🌿
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: "5.29.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Plant disease diagnosis using BLIP-2 + LangChain + Groq
---

# 🌿 PlantDoc AI

AI-powered plant disease diagnosis chatbot using:
- **BLIP-2** fine-tuned model for leaf image analysis
- **LangGraph ReAct agent** with memory
- **Groq (Llama 4 Scout)** as the LLM backbone
- **Tavily** for real-time web search
- **Smart Irrigation ML model**
- **Scientific PDF reference** for evidence-based answers

## 🔑 Required Secrets (set in HF Space Settings → Secrets)

| Secret Name | Description |
|---|---|
| `GROQ_API_KEY` | Get from [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | Get from [tavily.com](https://tavily.com) |
| `LANGSMITH_API_KEY` | (Optional) Get from [smith.langchain.com](https://smith.langchain.com) |
| `BLIP2_MODEL_ID` | HF Hub model ID for your BLIP-2 model (e.g. `your-username/blip2-plant-disease`) |

## 📁 Optional Files (upload to Space Files)

- `smart_irrigation_model.pkl` — irrigation ML model
- `smart_irrigation_preprocessor.pkl` — irrigation preprocessor
- `reference.pdf` — scientific reference book

## 🚀 How to Use

1. Upload a plant leaf image
2. (Optionally) set temperature, humidity, soil moisture
3. Click **إرسال** or pick a question from the category buttons
4. The agent will diagnose the disease, search references, and recommend treatment

## 🔧 Flutter Integration

Call the Space via Gradio's REST API:

```
POST https://your-username-plantdoc-ai.hf.space/api/predict
```

See the **Flutter Integration Guide** in the Files tab.
