"""
Comprehensive branching-story orchestrator with:
 • World context injection
 • Multi-turn Gemini conversations
 • Prompt caching
 • Parallel branch execution per stage
 • Eight predefined endings
 • Thriller & espionage flavor baked into prompts

Python 3.12.6 — 2025-06-13
"""

import os
import json
import time
import textwrap
import threading
import requests
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────── Configuration ───────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    # Fallback to temporary demo key when no environment variable present
    # This key will be revoked after the session
    API_KEY = "AIzaSyDMnUY_ivHVg46x5fM9tYAQ2gCxrvVof8M"
    print("[WARN] GEMINI_API_KEY not found; using temporary demo key")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-pro-2.5-06-05:generateContent"
)

HEADERS = {"Content-Type": "application/json"}
RUN_DIR = Path("./story_runs") / time.strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = RUN_DIR / "story_cache.json"
PROMPT_CACHE = RUN_DIR / "prompt_cache.json"

# Load world context
WORLD_CTX_PATH = Path("./WORLD_CONTEXT.md")
WORLD_CONTEXT = (
    WORLD_CTX_PATH.read_text(encoding="utf-8") if WORLD_CTX_PATH.exists() else ""
)

# ─────────────────── Master Prompts ───────────────────
SYSTEM_PROMPT = textwrap.dedent(
    """
You are a hybrid of Henry Kissinger (scholarly strategic analysis)
and John le Carré (spy‐thriller suspense). Write with:
  • geopolitical rigor
  • clandestine tradecraft detail
  • psychological tension
  • layered conspiracies & cliff-hangers

Prohibitions:
  • No supernatural or quantum miracles
  • No robotic or clichéd dialogue
  • Tech advances stay within plausible AI/AGI
"""
).strip()

STORY_OVERVIEW = textwrap.dedent(
    f"""
WORLD CONTEXT:
{WORLD_CONTEXT}

STORY SCOPE:
• Timeline: Jul 2025 – Dec 2030  
• Five stages:
    1. Jul 2025–Dec 2026 (6 × 3-mo slices)
    2. Jan 2027–Dec 2027 (4 × 3-mo)
    3. Jan 2028–Dec 2028 (4) → split 2 paths
    4. Jan 2029–Dec 2029 (4) → each splits
    5. Jan 2030–Dec 2030 (4) → 8 endings

• Endings (exactly eight):
    – 4 × Chinese strategic victory  
    – 2 × runaway AGI arms-race → global catastrophe  
    – 1 × AGI-safety détente  
    – 1 × narrow U.S. victory  
• Themes: espionage, AI/AGI race, economic coercion, proxy wars
"""
).strip()

KEMPEITAI_PROMPT = textwrap.dedent(
    """
Produce a classified intelligence dossier on the ‘Kempeitai Revival Society,’ including:
  • Esoteric ideology, ritual code, sources of fanaticism  
  • Historic lineage: WWII roots → Yakuza links → AI-era resurgence  
  • Covert tradecraft: dead-drops, sleeper agents, art-auction fund channels  
  • Key ops: 1998 Kuro-Gumo disinfo, 2014 Kiyomizu assassination,
             2025 NSA Yokota quantum-key heist  
  • Leadership: current ‘Shōgun-Kai’, inner council, foreign patrons  
  • Open questions that baffle allied agencies  
Style: IC Executive Summary + detailed analytic sections.
"""
).strip()

ITER_PROMPT_TEMPLATE = textwrap.dedent(
    """
CONTEXT:
{full_context}

TASK:
Continue this branch for {window} (Branch {branch_id}):
  • Add clandestine ops, spy-craft detail, double-agents, leaks  
  • Include mind-games, psychological pressure, surprise reveal  
  • End on a cliff-hanger or unanswered question  
  • Keep tech advances within plausible AI/AGI
Length: ~2,500–3,000 words.
"""
).strip()

# ───────────────── Data Structures ─────────────────
class Branch:
    def __init__(self, branch_id: str, ending: str):
        self.id = branch_id
        self.ending = ending
        self.conversation: List[Dict[str, str]] = []
        self.iterations: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ending": self.ending,
            "conversation": self.conversation,
            "iterations": self.iterations,
        }

ENDING_MAP = {
    "5.a.i.α": "PRC_WIN_1",
    "5.a.i.β": "PRC_WIN_2",
    "5.a.ii.α": "PRC_WIN_3",
    "5.a.ii.β": "PRC_WIN_4",
    "5.b.i.α": "AGI_DOOM_1",
    "5.b.i.β": "AGI_DOOM_2",
    "5.b.ii.α": "AGI_SAFETY",
    "5.b.ii.β": "US_WIN_NARROW",
}

STAGES = [
    ("Jul 2025–Dec 2026", 6),
    ("Jan 2027–Dec 2027", 4),
    ("Jan 2028–Dec 2028", 4),
    ("Jan 2029–Dec 2029", 4),
    ("Jan 2030–Dec 2030", 4),
]

# ─────────────── Prompt & Cache Utilities ───────────────
_lock = threading.Lock()

def load_json(path: Path, default):
    return json.loads(path.read_text("utf-8")) if path.exists() else default

prompt_cache = load_json(PROMPT_CACHE, {})

def save_prompt_cache():
    with open(PROMPT_CACHE, "w", encoding="utf-8") as f:
        json.dump(prompt_cache, f, ensure_ascii=False, indent=2)

def call_gemini_multi(messages: List[Dict[str, str]]) -> str:
    user_msg = messages[-1]["content"]
    if user_msg in prompt_cache:
        return prompt_cache[user_msg]
    payload = {
        "contents": [{"role": "system", "parts": [{"text": SYSTEM_PROMPT}]}]
        + [{"role": m["role"], "parts": [{"text": m["content"]}]} for m in messages],
        "generationConfig": {
            "maxOutputTokens": 4096,
            "temperature": 0.8,
            "topP": 0.95,
        },
    }
    resp = requests.post(f"{GEMINI_URL}?key={API_KEY}", headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    out = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    with _lock:
        prompt_cache[user_msg] = out
        save_prompt_cache()
    return out

# ─────────────── Story Orchestrator ───────────────
class StoryOrchestrator:
    def __init__(self):
        self.branches: Dict[str, Branch] = {
            bid: Branch(bid, ending) for bid, ending in ENDING_MAP.items()
        }
        if CACHE_PATH.exists():
            data = load_json(CACHE_PATH, {})
            for bid, bdata in data.get("branches", {}).items():
                b = self.branches[bid]
                b.conversation = bdata["conversation"]
                b.iterations = bdata["iterations"]

    def save_state(self):
        state = {"branches": {bid: br.to_dict() for bid, br in self.branches.items()}}
        CACHE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    def run(self):
        kemptxt = call_gemini_multi([{"role": "user", "content": KEMPEITAI_PROMPT}])
        kempeitai_header = f"### Classified Kempeitai Dossier\n{kemptxt}\n\n"
        for br in self.branches.values():
            if not br.conversation:
                br.conversation.append({"role": "assistant", "content": STORY_OVERVIEW})
                br.conversation.append({"role": "assistant", "content": kempeitai_header})

        for stage_idx, (label, count) in enumerate(STAGES, start=1):
            depth = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4}[stage_idx]
            alive = [b for b in self.branches.values() if len(b.id.split(".")) == depth]
            print(f"\n=== Stage {stage_idx}: {label} (branches: {len(alive)}) ===")
            with ThreadPoolExecutor(max_workers=4) as exe:
                futures = []
                for br in alive:
                    for i in range(count):
                        if i < len(br.iterations):
                            continue
                        window = f"{label} slice {i+1}/{count}"
                        full_ctx = "".join(m["content"] for m in br.conversation)
                        user_prompt = ITER_PROMPT_TEMPLATE.format(
                            full_context=full_ctx, window=window, branch_id=br.id
                        )
                        br.conversation.append({"role": "user", "content": user_prompt})
                        futures.append(
                            (br, i, exe.submit(call_gemini_multi, list(br.conversation)))
                        )
                for br, idx, fut in futures:
                    text = fut.result()
                    br.conversation.append({"role": "assistant", "content": text})
                    br.iterations.insert(idx, text)
                    print(f"✔ Branch {br.id} slice {idx+1}/{count}")
                    self.save_state()
        print(f"\nAll done. Story state in {CACHE_PATH}")

if __name__ == "__main__":
    orchestrator = StoryOrchestrator()
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("Interrupted; state saved.")
        orchestrator.save_state()
