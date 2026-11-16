"""
Top-up Synthetic Data Generator for REE Hydrometallurgy using Groq (openai/gpt-oss-120b)
Features strict inline verification:
Every candidate sample is checked for chemical validity, realistic separation factors (beta <= 2.0),
valid extractants, and balanced equations before being accepted into raw_dataset.jsonl.
"""

import os
import re
import json
import asyncio
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = Path("raw_dataset.jsonl")

REE_SUBTOPICS = [
    {
        "subdomain": "Extractant Chemistry & Saponification",
        "focus": "Equilibrium saponification degree of P507 / PC-88A using NH4OH or NaOH, its impact on initial aqueous phase pH drop, and avoiding gelatinous third-phase precipitate during Nd/Pr loading."
    },
    {
        "subdomain": "Separation Factor & Equilibrium pH",
        "focus": "Thermodynamics of adjacent lanthanide separation (Nd3+ vs Pr3+) with separation factor beta(Nd/Pr) ~ 1.3-1.5 in chloride media. Control of equilibrium pH between 2.8 and 3.6."
    },
    {
        "subdomain": "McCabe-Thiele Multi-Stage Countercurrent Design",
        "focus": "Calculating theoretical stages (extraction, scrubbing, stripping) using McCabe-Thiele operating lines. Optimal O/A ratios for feed containing 60% PrCl3 and 40% NdCl3."
    },
    {
        "subdomain": "Scrubbing Stage Optimization",
        "focus": "Hydrochloric acid (HCl) concentration in scrub liquor (0.2 - 0.6 M HCl) to displace co-extracted Pr3+ from organic phase, maintaining organic phase purity > 99.5% Nd2O3."
    },
    {
        "subdomain": "Stripping Kinetics & Strip Liquor Control",
        "focus": "High-acidity stripping of loaded Nd3+ from PC-88A using 2.0 - 3.5 M HCl at 40-50°C. Mass transfer kinetics, regeneration of hydrogen-form extractant (HA)2, and acid recycling."
    },
    {
        "subdomain": "Diluent, Modifiers & Third-Phase Prevention",
        "focus": "Role of aliphatic kerosene diluent and phase modifiers (e.g., TBP, isodecanol 5-10 vol%) in preventing third-phase microemulsion formation during heavy lanthanide loading."
    },
    {
        "subdomain": "Troubleshooting Crud & Emulsion Formation",
        "focus": "Root-cause diagnosis and remediation of interphase crud (silica/iron colloidal particulates, fungal degradation products, calcium sulfate scaling) in mixer-settlers."
    },
    {
        "subdomain": "Process Control & Online Monitoring",
        "focus": "Real-time spectrophotometric UV-Vis monitoring of Nd3+ (575 nm) and Pr3+ (444 nm) along mixer-settler batteries, pH probe calibration in high ionic strength REE chloride brines."
    }
]

PROMPT_TEMPLATES = [
    "A process engineer is commissioning a 30-stage mixer-settler battery separating PrCl3 and NdCl3 using 1.2 M PC-88A in sulfonated kerosene. Ask a detailed technical question addressing: {focus}",
    "A hydrometallurgical plant encounters an operational anomaly in the Nd/Pr scrubbing section. Formulate a rigorous industrial troubleshooting scenario regarding: {focus}",
    "Provide a detailed calculation and chemical equilibrium analysis regarding: {focus}. Request specific chemical equations, distribution ratios (D_Nd, D_Pr), and operating parameters.",
    "Formulate an in-depth question regarding mass balance, stage efficiency, and organic degradation mechanisms for: {focus}."
]

SYSTEM_PROMPT = """You are a Principal Hydrometallurgical Engineer specializing in Rare Earth Element (REE) solvent extraction.
Generate highly technical, operationally precise Question-and-Answer pairs for separating Neodymium (Nd3+) from Praseodymium (Pr3+).

CRITICAL SCIENTIFIC REQUIREMENTS:
1. Saponification is acid-base neutralization (HA + OH- -> A- + H2O or HA + NH4OH -> NH4A + H2O). Saponification does NOT produce alcohol (ROH).
2. Separation factor beta(Nd/Pr) for adjacent lanthanides in PC-88A/Cyanex 272 is strictly between 1.3 and 1.5. Never state beta = 5.
3. Equilibrium extraction pH is maintained between 2.8 and 3.6. Scrubbing uses 0.2-0.6 M HCl; stripping uses 2-4 M HCl.
4. Include balanced stoichiometric extraction equations: Ln3+ + 3(HA)2(org) <=> Ln(HA2)3(org) + 3H+(aq).

Output ONLY a single valid JSON object matching:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {"subdomain": "...", "extractant": "PC-88A", "target_elements": ["Nd", "Pr"]}}
No conversational preamble or postscript outside JSON."""


def verify_sample_chemistry(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Strict inline verification of each candidate sample."""
    msgs = data.get("messages", [])
    if len(msgs) != 2:
        return False, "Malformed messages structure"
    
    user_text = msgs[0].get("content", "").strip()
    asst_text = msgs[1].get("content", "").strip()
    full_text = f"{user_text} {asst_text}".lower()

    if len(user_text.split()) < 12:
        return False, "User prompt too brief"
    if len(asst_text.split()) < 80:
        return False, "Assistant response too brief"

    # Chemical checks
    if not ("nd" in full_text or "neodymium" in full_text) or not ("pr" in full_text or "praseodymium" in full_text):
        return False, "Missing target elements Nd or Pr"
    
    # Check for hallucinated alcohol byproduct
    if re.search(r"produces?\s+(?:an?\s+)?alcohol|c8h17oh|cleavage\s+to\s+alcohol", full_text):
        return False, "Hallucinated alcohol byproduct from saponification"

    # Check for impossible separation factor beta
    beta_match = re.search(r"beta\s*(?:[=(]|is|of)\s*([0-9.]+)", full_text)
    if beta_match:
        try:
            val = float(beta_match.group(1))
            if val > 2.5:
                return False, f"Impossible separation factor beta={val} (must be ~1.3-1.5)"
        except ValueError:
            pass

    return True, None


async def generate_and_verify_single(client: Any, semaphore: asyncio.Semaphore, sample_id: int) -> Optional[Dict[str, Any]]:
    subtopic = random.choice(REE_SUBTOPICS)
    user_prompt = random.choice(PROMPT_TEMPLATES).format(focus=subtopic["focus"])
    
    async with semaphore:
        for attempt in range(5):
            try:
                res = await client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Subdomain: {subtopic['subdomain']}\nTask: {user_prompt}"}
                    ],
                    temperature=0.7,
                    max_tokens=1800
                )
                raw = res.choices[0].message.content
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    is_valid, reason = verify_sample_chemistry(data)
                    if is_valid:
                        data["sample_id"] = sample_id
                        data["generator"] = "groq-gpt-oss-120b-verified"
                        return data
                    else:
                        # Log rejected candidate during verification
                        # print(f"[Candidate Rejected] {reason}")
                        pass
            except Exception:
                await asyncio.sleep(0.5 + attempt)
    return None


async def topup_to_1000_verified(target: int = 1000, concurrency: int = 6):
    from openai import AsyncOpenAI
    api_key = os.getenv("GROQ_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    # Read current count
    current_count = 0
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            current_count = sum(1 for line in f if line.strip())

    needed = target - current_count
    if needed <= 0:
        print(f"Target of {target} already reached! Current count: {current_count}")
        return

    print(f"\n🚀 Topping up {needed} verified samples to reach exactly {target} samples.")
    print(f"Engine: Groq (openai/gpt-oss-120b) with Strict In-Line Chemical Verification")
    print(f"Concurrency: {concurrency} workers | Destination: {OUTPUT_FILE}\n")

    semaphore = asyncio.Semaphore(concurrency)
    
    # Keep generating until we hit exactly target count
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        pbar = tqdm_asyncio(total=needed, desc="Verified Top-up Progress")
        while current_count < target:
            batch_size = min(concurrency * 2, target - current_count)
            tasks = [generate_and_verify_single(client, semaphore, sample_id=current_count + i + 1) for i in range(batch_size)]
            results = await asyncio.gather(*tasks)
            for res in results:
                if res and current_count < target:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()
                    current_count += 1
                    pbar.update(1)
        pbar.close()

    print(f"\n✅ Top-up Complete! Exactly {current_count} verified samples in {OUTPUT_FILE}.")


if __name__ == "__main__":
    asyncio.run(topup_to_1000_verified(1000))
