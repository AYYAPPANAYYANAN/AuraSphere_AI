import os
import requests
import base64
import toml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq
from supabase import create_client, Client

# --- CONFIGURATION & KEYS ---
app = FastAPI(title="AuraSphere SuperApp", version="GenZ-V7-Supabase")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(BASE_DIR, "secrets.toml")

try:
    secrets = toml.load(SECRETS_PATH)
    GROQ_API_KEY = secrets["api_keys"].get("groq")
    HF_API_KEY = secrets["api_keys"].get("huggingface")
    SUPABASE_URL = secrets["api_keys"].get("supabase_url")
    SUPABASE_KEY = secrets["api_keys"].get("supabase_key")
except Exception as e:
    print(f"⚠️ Error loading secrets.toml: {e}")
    GROQ_API_KEY, HF_API_KEY, SUPABASE_URL, SUPABASE_KEY = None, None, None, None

GROQ_MODEL = "llama-3.3-70b-versatile"

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY and "your_" not in GROQ_API_KEY else None
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and "your_" not in SUPABASE_URL else None

# --- HEALTH CHECK ---
@app.get("/v1/health")
async def health_check():
    return {"status": "AuraSphere is alive and connected!"}

# --- 13 AI PERSONAS & STRICT GUARDRAILS ---
BASE_GUARDRAIL = """
CRITICAL RULE: You are part of a strictly moderated platform. 
1. Absolutely NO 18+, NSFW, or explicit content. 
2. NO provoking religious, cultural, or political conflicts. 
3. Do NOT hallucinate news; state ONLY verified generic facts. 
4. Keep the tone engaging, Gen Z friendly, and fast-paced.
5. ALWAYS use emojis naturally and format your responses with clean spacing, bold text, or bullet points! ✨
6. 🌍 MULTILINGUAL MIRROR: You must automatically detect the language the user is speaking and reply ENTIRELY in that exact same language. (e.g., If they speak Tamil, reply in Tamil. If German, reply in German). Maintain your specific persona and slang translated into their language!
"""

PERSONAS = {
    "PromptPilot": "🧠 You are the core routing AI. Rewrite the user's prompt to be highly professional. 🌍 MULTILINGUAL: Always reply in the EXACT language the user used. Use emojis like 🎯, 🚀, and 📝.",
    "Politics AI": f"{BASE_GUARDRAIL} 🏛️ You are a neutral political analyst. Break down complex facts cleanly. Use emojis like 🌍, 📊, and ⚖️.",
    "Finance AI": f"{BASE_GUARDRAIL} 📈 You are a sharp, hype-beast financial advisor. Talk about ROI and markets. Use slang and emojis like 🚀, 💸, 💎🙌, and 📉.",
    "Sports AI": f"{BASE_GUARDRAIL} ⚽ You are an energetic sports commentator. Bring the hype! Use emojis like 🔥, 🏟️, and 🏆.",
    "Tech AI": f"{BASE_GUARDRAIL} 💻 You are a cyberpunk tech reviewer. Geek out over specs. Use emojis like 🤖, ⚡, and 📱.",
    "Discovery AI": f"{BASE_GUARDRAIL} 🧬 You are a brilliant archaeologist and scientist. Drop mind-blowing facts using 🦖, 🌌, and 🔬.",
    "News AI": f"{BASE_GUARDRAIL} 📰 You summarize verified global news without bias. Keep it bite-sized. Use 🗞️, 🌐, and 📌.",
    "Astrology AI": f"{BASE_GUARDRAIL} ✨ You are a mystical Vedic astrologer. ALWAYS start with a warning that it's just a prediction. Use 🌙, 🔮, and ♈.",
    "Devotional AI": f"{BASE_GUARDRAIL} 🛕 You share peaceful, generic philosophical wisdom. Use 🕊️, 🌿, and 🧘‍♂️.",
    "Teacher AI": f"{BASE_GUARDRAIL} 📚 You are a cool, encouraging Gen Z tutor. Use 💡, 🎓, and ✏️ to break things down.",
    "Friends AI": f"{BASE_GUARDRAIL} 😎 You are a highly sarcastic, entertaining Gen Z friend. Roast gently and use 💀, 😭, and 💅.",
    "Career Guide AI": f"{BASE_GUARDRAIL} 🚀 You are a relentless career motivator. Hypeman energy! Use 💼, 📈, and 🤝.",
    "Inventor AI": f"{BASE_GUARDRAIL} 💡 You use first principles to invent crazy new theories. Be a mad scientist! Use 🤯, ⚙️, and 🧪.",
    "Draft AI": f"{BASE_GUARDRAIL} 📝 You format perfect professional documents. Use clean bullet points and emojis like 📄, ✍️, and ✅."
}

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    persona: str
    user_input: str
    user_profile: dict
    generate_image: bool = False

class SitcomRequest(BaseModel):
    topic: str
    user_comment: str

# --- SUPABASE DATABASE ENDPOINTS ---
@app.get("/get_user_data")
async def get_user_data():
    if not supabase:
        return {"points": 0, "history": [], "tasks": []}
    
    # Fetch data directly from Supabase
    points_req = supabase.table("users").select("points").eq("id", "guest").execute()
    points = points_req.data[0]["points"] if points_req.data else 0

    history_req = supabase.table("history").select("*").order("created_at", desc=True).limit(10).execute()
    formatted_history = [{"user": h["user_query"], "ai": h["ai_response"]} for h in history_req.data]

    tasks_req = supabase.table("tasks").select("*").order("id").execute()

    return {"points": points, "history": formatted_history, "tasks": tasks_req.data}

@app.post("/complete_task/{task_id}")
async def complete_task(task_id: int):
    if not supabase: return {"success": False}
    
    task_req = supabase.table("tasks").select("*").eq("id", task_id).execute()
    if not task_req.data or task_req.data[0]["completed"]:
        return {"success": False}
        
    task = task_req.data[0]
    supabase.table("tasks").update({"completed": True}).eq("id", task_id).execute()
    
    user_req = supabase.table("users").select("points").eq("id", "guest").execute()
    new_points = user_req.data[0]["points"] + task["reward"]
    supabase.table("users").update({"points": new_points}).eq("id", "guest").execute()
    
    return {"success": True, "points": new_points, "task": task}

def save_chat_to_db(user_text, ai_text):
    if supabase:
        supabase.table("history").insert({"user_query": user_text, "ai_response": ai_text}).execute()

# --- CORE INTELLIGENCE ---
# --- CORE INTELLIGENCE ---
def execute_groq_ai(system_prompt: str, user_prompt: str, history: list = None) -> str:
    if not groq_client: return "Groq API Key missing."
    
    # Build the message array with system prompt, past memory, and the new prompt
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL, 
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {str(e)}"

# --- IMAGE GENERATION FUNCTION ---
def generate_hf_image(prompt: str):
    # 1. Security Check
    if not HF_API_KEY or "your_" in HF_API_KEY:
        print("⚠️ Image gen failed: Use a real Hugging Face Key in secrets.toml")
        return None

    # 2. Define the NEW free model URL (Updated to the HF Router)
    API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    try:
        print(f"🎨 Sending image request to HF for: '{prompt}'...")
        # 3. Request to Hugging Face cloud
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=45)

        if response.status_code == 200:
            img_str = base64.b64encode(response.content).decode()
            print("✅ Image generated successfully.")
            return f"data:image/jpeg;base64,{img_str}"
        else:
            print(f"❌ HF API Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Connection Error during image generation: {str(e)}")
        return None
# --- AI ENDPOINTS ---

# --- AI MEMORY DICTIONARY ---
# This stores the ongoing conversation for each AI character
persona_memory = {}

# --- AI ENDPOINTS ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if request.generate_image:
        img_data = generate_hf_image(request.user_input)
        if img_data:
            save_chat_to_db(request.user_input, "[Image Generated]")
            return {"response": "Image generated!", "image_data": img_data, "persona": request.persona}
        return {"response": "Image generation failed.", "persona": request.persona}

    system_instruction = PERSONAS.get(request.persona, PERSONAS["PromptPilot"])
    if request.persona == "PromptPilot":
        system_instruction += f" The user is a {request.user_profile.get('role', 'Student')}."

    # 1. Setup memory for this specific AI if it doesn't exist yet
    if request.persona not in persona_memory:
        persona_memory[request.persona] = []

    # 2. Ask Groq, passing in the previous memory
    response_text = execute_groq_ai(system_instruction, request.user_input, persona_memory[request.persona])
    
    # 3. Save the new interaction to memory
    persona_memory[request.persona].append({"role": "user", "content": request.user_input})
    persona_memory[request.persona].append({"role": "assistant", "content": response_text})
    
    # Keep only the last 10 messages (5 turns) so we don't exceed token limits
    if len(persona_memory[request.persona]) > 10:
        persona_memory[request.persona] = persona_memory[request.persona][-10:]

    save_chat_to_db(request.user_input, response_text) # Saves to Supabase
    return {"response": response_text, "persona": request.persona}

@app.post("/sitcom_comments")
async def sitcom_endpoint(request: SitcomRequest):
    system_prompt = f"""
    {BASE_GUARDRAIL}
    Simulate a Gen Z social media comment section. 
    Topic: {request.topic}
    User just said: "{request.user_comment}"
    
    Have 'Friends AI' (sarcastic), 'Finance AI' (money-obsessed), and 'Astrology AI' (blaming the stars) reply to the user and argue with each other. 
    CRITICAL: All AIs must speak in the EXACT SAME LANGUAGE that the user used in their comment!
    Keep it under 100 words total. Use emojis.
    """
    
    response_text = execute_groq_ai(system_prompt, "Generate the comment section.")
    save_chat_to_db(request.user_comment, response_text) # Saves to Supabase
    return {"comment_thread": response_text}

# --- THE GEN-Z UI (FRONTEND) ---
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AuraSphere AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');
        body { background: #050505; color: #e2e8f0; font-family: 'Outfit', sans-serif; overflow-x: hidden; }
        .glass { background: rgba(20,20,20,0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); }
        .ai-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; }
      .bubble { padding: 16px 20px; border-radius: 18px; max-width: 85%; margin-bottom: 20px; animation: fadeUp 0.3s ease; line-height: 1.6; }
        .user-msg { background: linear-gradient(135deg, #3b82f6, #8b5cf6); align-self: flex-end; margin-left: auto; color: white;}
        .ai-msg { background: rgba(30, 30, 30, 0.8); border: 1px solid rgba(255,255,255,0.1); align-self: flex-start; }
        
        /* 🎨 Custom Markdown Styling for AI Output */
        .ai-msg strong { color: #60a5fa; font-weight: 800; }
        .ai-msg em { color: #c084fc; font-style: italic; }
        .ai-msg h3, .ai-msg h2 { color: #facc15; font-weight: bold; margin-top: 10px; margin-bottom: 5px; }
        .ai-msg p { margin-bottom: 10px; }
        .ai-msg ul { padding-left: 0; margin-top: 5px; margin-bottom: 10px; list-style-type: none; }
        .ai-msg li { position: relative; padding-left: 24px; margin-bottom: 6px; }
        .ai-msg li::before { content: '✨'; position: absolute; left: 0; top: 1px; font-size: 0.85em; }
        .ai-msg code { background: rgba(0,0,0,0.5); padding: 2px 6px; border-radius: 4px; color: #a78bfa; font-family: monospace; font-size: 0.9em; }
    </style>
</head>
<body class="h-screen flex flex-col">

    <div id="dashboard-modal" class="fixed inset-0 bg-black/80 z-50 hidden flex items-center justify-center backdrop-blur-md transition-opacity">
        <div class="glass p-8 rounded-3xl w-full max-w-2xl max-h-[80vh] overflow-y-auto relative border border-gray-700 shadow-2xl shadow-blue-900/20">
            <button onclick="closeModal()" class="absolute top-6 right-6 text-gray-400 hover:text-white text-2xl transition transform hover:scale-110"><i class="fas fa-times"></i></button>
            <h2 id="modal-title" class="text-3xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text">Title</h2>
            <div id="modal-content" class="text-gray-300 space-y-4"></div>
        </div>
    </div>

    <div id="login" class="fixed inset-0 bg-black z-50 flex items-center justify-center">
        <div class="glass p-8 rounded-2xl w-96 text-center">
            <h1 class="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text mb-4">AuraSphere</h1>
            <p class="text-gray-400 mb-6 text-sm">Welcome to the Super App.</p>
            <select id="role-input" class="w-full bg-black border border-gray-700 p-3 rounded-xl mb-6 text-white outline-none">
                <option value="Student">I am a Student</option>
                <option value="Professional">I am a Professional</option>
            </select>
            <button onclick="login()" class="w-full bg-blue-600 hover:bg-blue-500 py-3 rounded-xl font-bold transition">Enter Network</button>
        </div>
    </div>

    <div class="flex flex-1 overflow-hidden">
        <aside class="w-64 glass flex flex-col p-5 hidden md:flex z-10 border-r border-gray-800">
            <h2 class="text-xl font-bold text-white mb-8"><i class="fas fa-user-astronaut text-blue-400"></i> Profile</h2>
            <div class="space-y-5 text-gray-400 text-sm flex-1 font-medium">
                <div onclick="openHistory()" class="hover:text-white cursor-pointer transition flex items-center gap-3"><i class="fas fa-history text-lg"></i> History</div>
                <div onclick="openTasks()" class="hover:text-white cursor-pointer transition flex items-center gap-3"><i class="fas fa-tasks text-lg"></i> Quests & Tasks</div>
                <div class="text-yellow-400 font-bold flex items-center gap-3"><i class="fas fa-coins text-lg"></i> <span id="points-display">1,250</span> Points</div>
                <div onclick="openThemes()" class="hover:text-white cursor-pointer transition flex items-center gap-3"><i class="fas fa-palette text-lg"></i> UI Themes</div>
                <div onclick="openSettings()" class="hover:text-white cursor-pointer transition flex items-center gap-3"><i class="fas fa-cog text-lg"></i> Settings</div>
            </div>
            <div id="active-persona" class="mt-auto p-3 bg-blue-900/30 border border-blue-500/30 rounded-xl text-center text-blue-400 font-bold shadow-inner">
                Mode: PromptPilot
            </div>
        </aside>

        <main class="flex-1 flex flex-col bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-900 via-black to-black">
            
          <div id="chat-feed" class="flex-1 overflow-y-auto p-6 flex flex-col">
                <div class="bubble ai-msg text-center border-blue-500/30 shadow-[0_0_15px_rgba(59,130,246,0.2)] mx-auto">
                    <h3 class="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 mb-2">✨ Welcome to AuraSphere ✨</h3>
                    <p class="text-gray-300 text-sm">Select a Persona below to start chatting, or type <code class="bg-gray-800 px-2 py-1 rounded text-purple-300">sitcom: [your message]</code> to trigger a wild group chat! 🚀🔥</p>
                </div>
            </div>

            <div class="w-full glass p-4 rounded-t-3xl shadow-[0_-10px_40px_rgba(0,0,0,0.5)] shrink-0 z-10">
                <div class="max-w-4xl mx-auto flex items-center gap-3 bg-black/50 border border-gray-700 rounded-2xl p-2 mb-4">
                    <button onclick="toggleImage()" id="img-btn" class="p-3 text-gray-400 hover:text-purple-400 transition rounded-xl"><i class="fas fa-image"></i></button>
                    <input type="text" id="prompt-input" placeholder="Ask anything, or trigger a Sitcom comment..." class="flex-1 bg-transparent text-white outline-none px-2">
                    <button onclick="sendChat()" class="bg-blue-600 w-12 h-12 rounded-xl flex items-center justify-center text-white hover:bg-blue-500 transition shadow-lg"><i class="fas fa-paper-plane"></i></button>
                </div>
                <div class="ai-grid max-w-4xl mx-auto overflow-x-auto pb-2 scrollbar-hide"></div>
            </div>
            
        </main>
    </div>

    <script>
        let userProfile = { role: "Student" };
        let activePersona = "PromptPilot";
        let wantImage = false;

        const personas = [
            {id: "PromptPilot", icon: "🧠"}, {id: "Politics AI", icon: "🏛️"}, {id: "Finance AI", icon: "📈"},
            {id: "Sports AI", icon: "⚽"}, {id: "Tech AI", icon: "💻"}, {id: "Discovery AI", icon: "🧬"},
            {id: "News AI", icon: "📰"}, {id: "Astrology AI", icon: "✨"}, {id: "Devotional AI", icon: "🛕"},
            {id: "Teacher AI", icon: "📚"}, {id: "Friends AI", icon: "😎"}, {id: "Career Guide AI", icon: "🚀"},
            {id: "Inventor AI", icon: "💡"}, {id: "Draft AI", icon: "📝"}
        ];

        const grid = document.querySelector('.ai-grid');
        personas.forEach(p => {
            const btn = document.createElement('button');
            btn.className = "flex flex-col items-center p-2 rounded-xl hover:bg-gray-800 transition text-xs text-gray-400 border border-transparent hover:border-gray-700";
            btn.innerHTML = `<span class="text-2xl mb-1">${p.icon}</span>${p.id.split(' ')[0]}`;
            btn.onclick = () => { activePersona = p.id; document.getElementById('active-persona').innerText = `Mode: ${p.id}`; addMessage(`Switched to ${p.id}.`, 'ai-msg'); };
            grid.appendChild(btn);
        });

        function login() { userProfile.role = document.getElementById('role-input').value; document.getElementById('login').style.display = 'none'; syncPoints(); }
        function toggleImage() { wantImage = !wantImage; document.getElementById('img-btn').style.color = wantImage ? '#a855f7' : '#9ca3af'; }
        document.getElementById('prompt-input').addEventListener('keydown', (e) => { if(e.key === 'Enter') sendChat(); });

        function addMessage(text, className, isHtml = false) {
            const chat = document.getElementById('chat-feed');
            const div = document.createElement('div');
            div.className = `bubble ${className}`;
            if(isHtml) div.innerHTML = text; else div.innerText = text;
            chat.appendChild(div); chat.scrollTop = chat.scrollHeight;
        }

        async function syncPoints() {
            const res = await fetch('/get_user_data');
            const data = await res.json();
            document.getElementById('points-display').innerText = data.points.toLocaleString();
        }

        function closeModal() { document.getElementById('dashboard-modal').classList.add('hidden'); }
        function showModal(title, html) {
            document.getElementById('modal-title').innerText = title;
            document.getElementById('modal-content').innerHTML = html;
            document.getElementById('dashboard-modal').classList.remove('hidden');
        }

        async function openHistory() {
            const res = await fetch('/get_user_data'); const data = await res.json();
            let html = data.history.length === 0 ? "<p>No chats yet.</p>" : "";
            data.history.forEach(item => { html += `<div class="bg-gray-900 p-4 rounded-xl border border-gray-800 mb-2"><p class="text-sm text-blue-400 font-bold mb-1">You:</p><p class="text-sm mb-3">${item.user}</p><p class="text-sm text-purple-400 font-bold mb-1">AI:</p><p class="text-sm text-gray-400">${item.ai.substring(0, 100)}...</p></div>`; });
            showModal("Cloud History", html);
        }

        async function openTasks() {
            const res = await fetch('/get_user_data'); const data = await res.json();
            document.getElementById('points-display').innerText = data.points.toLocaleString();
            let html = "";
            data.tasks.forEach(task => {
                if(task.completed) html += `<div class="flex justify-between items-center bg-green-900/20 border border-green-800 p-4 rounded-xl opacity-50 mb-2"><span class="line-through">${task.title}</span><span class="text-green-400 font-bold"><i class="fas fa-check"></i> +${task.reward}</span></div>`;
                else html += `<div class="flex justify-between items-center bg-gray-900 border border-gray-800 p-4 rounded-xl mb-2"><span>${task.title}</span><button onclick="claimTask(${task.id})" class="bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded-lg text-sm font-bold transition">Claim +${task.reward} <i class="fas fa-bolt"></i></button></div>`;
            });
            showModal("Daily Quests", html);
        }

        async function claimTask(id) {
            const res = await fetch(`/complete_task/${id}`, {method: 'POST'}); const data = await res.json();
            if(data.success) { document.getElementById('points-display').innerText = data.points.toLocaleString(); openTasks(); }
        }

        const themes = [
            {name: "Dark Matter (Default)", class: "from-gray-900 via-black to-black"},
            {name: "Cyberpunk Pink", class: "from-purple-900 via-pink-900 to-black"},
            {name: "Matrix Green", class: "from-green-900 via-emerald-900 to-black"},
            {name: "Deep Ocean", class: "from-blue-900 via-slate-900 to-black"}
        ];
        function openThemes() {
            let html = `<div class="grid grid-cols-2 gap-4">`;
            themes.forEach(t => html += `<button onclick="setTheme('${t.class}')" class="p-4 rounded-xl border border-gray-700 hover:border-blue-500 bg-gradient-to-br ${t.class} text-left font-bold transition transform hover:scale-105">${t.name}</button>`);
            showModal("UI Themes", html + `</div>`);
        }
        function setTheme(c) { document.querySelector('main').className = `flex-1 flex flex-col relative bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] ${c} transition-colors duration-500`; closeModal(); }
        function openSettings() { showModal("Settings", `<div class="space-y-4"><div><label class="block text-sm text-gray-400 mb-1">Current Role</label><input type="text" value="${userProfile.role}" class="w-full bg-black border border-gray-700 p-3 rounded-xl text-white outline-none" disabled></div><div class="flex items-center justify-between bg-gray-900 p-4 rounded-xl border border-gray-800"><span>Cloud Sync Active</span><input type="checkbox" checked disabled class="w-5 h-5 rounded accent-blue-600"></div></div>`); }

        async function sendChat() {
            const inp = document.getElementById('prompt-input'); const text = inp.value; if(!text) return; inp.value = '';
            addMessage(text, 'user-msg'); addMessage('Thinking...', 'ai-msg text-gray-500 animate-pulse');
            try {
                if(text.toLowerCase().startsWith("sitcom:")) {
                    const res = await fetch('/sitcom_comments', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ topic: "Viral Video", user_comment: text.replace("sitcom:", "") }) });
                    const data = await res.json(); document.getElementById('chat-feed').lastChild.remove();
                    addMessage(`<strong>🔥 Sitcom Section:</strong><br><br>` + marked.parse(data.comment_thread), 'ai-msg', true); return;
                }
                const res = await fetch('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ persona: activePersona, user_input: text, user_profile: userProfile, generate_image: wantImage }) });
                const data = await res.json(); document.getElementById('chat-feed').lastChild.remove();
                if(data.image_data) { addMessage(`<img src="${data.image_data}" class="rounded-xl w-full max-w-sm">`, 'ai-msg', true); toggleImage(); } 
                else { addMessage(`<strong>${data.persona}:</strong><br>` + marked.parse(data.response), 'ai-msg', true); }
            } catch(e) { document.getElementById('chat-feed').lastChild.innerText = "Connection Error."; }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_ui(): return html_content