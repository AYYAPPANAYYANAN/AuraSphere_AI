# ✨ AuraSphere SuperApp (GenZ-V7-Supabase)

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)
![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=flat-square&logo=supabase)
![Groq Engine](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-orange?style=flat-square)

Welcome to **AuraSphere**, an ultra-engaging, fast-paced FastAPI SuperApp built for Gen-Z workflows. It bundles 13 distinct AI conversational personas, cross-border multilingual mirroring, a gamified daily quest system hooked directly into **Supabase**, text-to-image synthesis, and a modular Tailwind CSS dashboard.

---

## 🔥 Key Core Features

* **🎭 13 Swappable AI Personas**: Jump between distinct modular personalities on the fly (e.g., `PromptPilot`, `Finance AI` hypebeast tracking ROI, sarcastic `Friends AI`, mad scientist `Inventor AI`, and more).
* **🌍 Multilingual Mirroring**: Automatic user language detection and translation fallback. If the request is in Tamil, German, or Japanese, the underlying system shifts entirely to that idiom while preserving persona slang.
* **⚡ Sitcom Mode**: Trigger an algorithmic comment thread simulation where multiple sub-agents (`Friends AI`, `Finance AI`, `Astrology AI`) argue with each other dynamically under a single context topic.
* **🎨 Visual Cortex Synth**: Toggle image generation on/off directly from the prompt bar using the routed **Stable Diffusion XL** backend via Hugging Face Inference.
* **📈 Gamified Supabase State Syncing**: Full support for fetching profile data, maintaining global conversational memory pools, tracking live interactive points, and updating daily gamified tasks.

---

## 🛠️ Architecture Stack

* **Backend Matrix**: FastAPI & Pydantic Validation Layers
* **Cognitive Processing**: `llama-3.3-70b-versatile` via Groq Cloud Core
* **Visual Synthesis**: `stable-diffusion-xl-base-1.0` via Hugging Face Router endpoints
* **Persistence Layer**: Supabase Client Ecosystem (Table relationships: `users`, `history`, `tasks`)
* **Responsive UI Layout**: Single-File Embedded Glassmorphism Web App (Tailwind CSS, Marked.js Markdown Engine, FontAwesome icons)

---

## 🚀 Setup & Local Deployment

### 1. Project Pre-requisites
Make sure you have Python 3.9 or higher running natively inside your virtual environment.

### 2. Clone Sub-System
Install Dependencies
Bash
pip install fastapi uvicorn requests python-dotenv pydantic groq supabase toml

