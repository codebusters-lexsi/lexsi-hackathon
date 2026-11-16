"""
Step 2: Asynchronous Synthetic Data Generation for Rare Earth Element (REE) Hydrometallurgy
Primary Engine: Cohere Labs Command R+ (with Groq fallback)
Domain: Solvent extraction parameters, equilibrium pH balancing, and troubleshooting 
for separating closely related lanthanides (Neodymium Nd3+ from Praseodymium Pr3+).
Outputs: raw_dataset.jsonl (Hugging Face conversational messages format)
"""

import os
import sys
import json
import asyncio
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = Path("raw_dataset.jsonl")

# Exhaustive technical seed topics covering Nd/Pr solvent extraction hydrometallurgy
REE_SUBTOPICS = [
    {
        "subdomain": "Extractant Chemistry & Saponification",
        "focus": "Equilibrium saponification degree of P507 / PC-88A (2-ethylhexylphosphonic acid mono-2-ethylhexyl ester) or Cyanex 272 using NH4OH or NaOH, its impact on initial aqueous phase pH drop, and avoiding gelatinous third-phase precipitate during Nd/Pr loading."
    },
    {
        "subdomain": "Separation Factor & Equilibrium pH",
        "focus": "Thermodynamics of adjacent lanthanide separation (Nd3+ vs Pr3+) with separation factor beta(Nd/Pr) ~ 1.3-1.5 in chloride media. Control of equilibrium pH between 2.8 and 3.6 to maximize separation efficiency while preventing premature stripping or precipitation."
    },
    {
        "subdomain": "McCabe-Thiele Multi-Stage Countercurrent Design",
        "focus": "Calculating theoretical stages (extraction, scrubbing, stripping) using McCabe-Thiele operating lines. Determining optimal Organic to Aqueous (O/A) ratios for feed containing 60% PrCl3 and 40% NdCl3."
    },
    {
        "subdomain": "Scrubbing Stage Optimization",
        "focus": "Hydrochloric acid (HCl) concentration in scrub liquor (0.2 - 0.6 M HCl) to displace co-extracted Pr3+ from the organic phase without excessively back-extracting Nd3+, maintaining organic phase purity > 99.5% Nd2O3."
    },
    {
        "subdomain": "Stripping Kinetics & Strip Liquor Control",
        "focus": "High-acidity stripping of loaded Nd3+ from PC-88A using 2.0 - 3.5 M HCl at 40-50°C. Mass transfer kinetics, regeneration of hydrogen-form extractant (HA)2, and acid recycling."
    },
    {
        "subdomain": "Diluent, Modifiers & Third-Phase Prevention",
        "focus": "Role of aliphatic kerosene diluent and phase modifiers (e.g., TBP, isodecanol 5-10 vol%) in preventing third-phase microemulsion formation during heavy lanthanide loading at high extractant concentrations (1.0 - 1.5 M)."
    },
    {
        "subdomain": "Troubleshooting Crud & Emulsion Formation",
        "focus": "Root-cause diagnosis and remediation of interphase crud (silica/iron colloidal particulates, fungal degradation products, calcium sulfate scaling) in mixer-settlers causing phase entrainment and organic loss."
    },
    {
        "subdomain": "Process Control & Online Monitoring",
        "focus": "Real-time spectrophotometric UV-Vis monitoring of Nd3+ (575 nm absorbance) and Pr3+ (444 nm absorbance) along mixer-settler batteries, pH probe calibration in high ionic strength REE chloride brines, and automated caustic dosing."
    },
    {
        "subdomain": "Comparative Extractant Systems",
        "focus": "Quantitative comparison between PC-88A (HEHEHP), Cyanex 272 (diisooctylphosphinic acid), and Versatic 10 for Nd/Pr separation: extraction constant Kex, pKa differences, stripping ease, and operational cost tradeoffs."
    },
    {
        "subdomain": "Hydrodynamic Instabilities & Phase Inversion",
        "focus": "Mixer-settler impeller tip speed, Weber number, organic-continuous vs aqueous-continuous operation, settler coalescing plate fouling, and preventing organic entrainment in the aqueous raffinate."
    }
]

PROMPT_TEMPLATES = [
    "A process engineer is commissioning a 30-stage mixer-settler battery separating PrCl3 and NdCl3 using 1.2 M PC-88A in sulfonated kerosene. Ask a detailed technical question addressing: {focus}",
    "A hydrometallurgical plant encounters an operational anomaly in the Nd/Pr scrubbing section. Formulate a rigorous industrial troubleshooting scenario regarding: {focus}",
    "Provide a detailed calculation and chemical equilibrium analysis regarding: {focus}. Request specific chemical equations, distribution ratios (D_Nd, D_Pr), and operating parameters.",
    "Draft a senior metallurgical inquiry examining pilot-scale versus commercial-scale deviations in: {focus}. Include temperature effects and acid consumption economics.",
    "Formulate an in-depth question regarding mass balance, stage efficiency, and organic degradation mechanisms for: {focus}."
]

SYSTEM_PROMPT = """You are a Principal Hydrometallurgical Engineer specializing in Rare Earth Element (REE) solvent extraction (SX).
Your task is to generate highly technical, rigorous, and operationally precise Question-and-Answer pairs for separating closely related light/middle lanthanides, specifically Neodymium (Nd3+) from Praseodymium (Pr3+).

Ensure every technical response includes:
1. Exact chemical species, extractant dimers (e.g., [HA]2, [H2A2]), extraction equilibria equations (e.g., Ln3+ + 3(HA)2(org) <=> Ln(HA2)3(org) + 3H+(aq)).
2. Quantitative parameters: equilibrium pH ranges (typically 2.8 - 3.6), separation factors beta(Nd/Pr) ~ 1.3 - 1.5, saponification fractions (30 - 45%), O/A phase ratios, HCl molarities (0.2-0.5M for scrubbing, 2-4M for stripping).
3. Physical and hydrodynamic phenomena: mixer-settler residence times, coalescer design, third-phase formation, diluent selection (aliphatic kerosene), phase modifier dosing (isodecanol/TBP), and interphase crud remediation.
4. Deep, actionable troubleshooting steps with no hand-waving or generic filler.

You must output strictly valid JSON matching this schema:
{
  "messages": [
    {"role": "user", "content": "<The complex technical inquiry>"},
    {"role": "assistant", "content": "<The comprehensive, mathematically and chemically precise engineering response>"}
  ],
  "metadata": {
    "subdomain": "<subdomain name>",
    "target_elements": ["Nd", "Pr"],
    "extractant": "<e.g. PC-88A / Cyanex 272 / P507>",
    "difficulty": "expert"
  }
}"""


async def generate_sample_cohere(
    co_client: Any,
    subtopic: Dict[str, str],
    user_prompt: str,
    model: str = "command-r-plus-08-2024"
) -> Dict[str, Any]:
    """Generates a sample using Cohere Labs V2 API."""
    res = await co_client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Subdomain: {subtopic['subdomain']}\nTask prompt: {user_prompt}"}
        ],
        response_format={"type": "json_object"}
    )
    raw_text = res.message.content[0].text
    return json.loads(raw_text)


async def generate_sample_groq(
    groq_client: Any,
    subtopic: Dict[str, str],
    user_prompt: str,
    model: str = "llama-3.3-70b-versatile"
) -> Dict[str, Any]:
    """Generates a sample using Groq OpenAI-compatible API."""
    res = await groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Subdomain: {subtopic['subdomain']}\nTask prompt: {user_prompt}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.75,
        max_tokens=1800
    )
    raw_text = res.choices[0].message.content
    return json.loads(raw_text)


async def generate_single_sample(
    co_client: Optional[Any],
    groq_client: Optional[Any],
    semaphore: asyncio.Semaphore,
    sample_id: int,
    prefer_provider: str = "cohere",
    cohere_model: str = "command-r-plus-08-2024",
    groq_model: str = "llama-3.3-70b-versatile"
) -> Optional[Dict[str, Any]]:
    """Generates one high-depth REE hydrometallurgy Q&A pair with automatic provider failover."""
    subtopic = random.choice(REE_SUBTOPICS)
    prompt_template = random.choice(PROMPT_TEMPLATES)
    user_prompt = prompt_template.format(focus=subtopic["focus"])

    max_retries = 4
    base_delay = 1.0

    async with semaphore:
        for attempt in range(max_retries):
            try:
                data = None
                # Try preferred provider first
                if prefer_provider == "cohere" and co_client:
                    try:
                        data = await generate_sample_cohere(co_client, subtopic, user_prompt, cohere_model)
                    except Exception as co_err:
                        # Failover to Groq if Cohere hits rate limit or error
                        if groq_client:
                            # print(f"Cohere retry/failover to Groq: {co_err}")
                            data = await generate_sample_groq(groq_client, subtopic, user_prompt, groq_model)
                        else:
                            raise co_err
                elif prefer_provider == "groq" and groq_client:
                    try:
                        data = await generate_sample_groq(groq_client, subtopic, user_prompt, groq_model)
                    except Exception as groq_err:
                        if co_client:
                            data = await generate_sample_cohere(co_client, subtopic, user_prompt, cohere_model)
                        else:
                            raise groq_err

                # Validate conversational schema
                if data and "messages" in data and len(data["messages"]) == 2:
                    data["sample_id"] = sample_id
                    data["generator"] = prefer_provider
                    return data
                else:
                    raise ValueError("JSON does not contain standard 2-turn messages array")

            except Exception as e:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0.1, 0.4)
                if attempt == max_retries - 1:
                    print(f"\n[Warning] Sample {sample_id} failed: {e}")
                    return None
                await asyncio.sleep(wait_time)
        return None


async def run_batch_generation(
    total_samples: int = 1000,
    concurrency: int = 5,
    output_path: Path = OUTPUT_FILE,
    prefer_provider: str = "cohere",
    cohere_model: str = "command-r-plus-08-2024",
    groq_model: str = "llama-3.3-70b-versatile"
):
    """Orchestrates async generation of total_samples and streams to raw_dataset.jsonl."""
    co_api_key = os.getenv("COHERE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    co_client = None
    if co_api_key:
        import cohere
        co_client = cohere.AsyncClientV2(api_key=co_api_key)

    groq_client = None
    if groq_api_key:
        from openai import AsyncOpenAI
        groq_client = AsyncOpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

    if not co_client and not groq_client:
        raise ValueError("Neither COHERE_API_KEY nor GROQ_API_KEY is available in .env.")

    semaphore = asyncio.Semaphore(concurrency)

    # Check for existing samples to allow resuming seamlessly
    existing_count = 0
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_count = sum(1 for line in f if line.strip())
        print(f"Detected existing {output_path} with {existing_count} samples. Resuming...")

    remaining = total_samples - existing_count
    if remaining <= 0:
        print(f"Dataset already complete with {existing_count} samples at {output_path}.")
        return

    print(f"\n🧪 Generating {remaining} technical REE Hydrometallurgy Q&A pairs (Target: {total_samples})")
    print(f"Primary Provider: Cohere Labs ({cohere_model}) | Secondary: Groq ({groq_model})")
    print(f"Concurrency: {concurrency} workers | Destination: {output_path}\n")

    tasks = [
        generate_single_sample(
            co_client=co_client,
            groq_client=groq_client,
            semaphore=semaphore,
            sample_id=existing_count + i + 1,
            prefer_provider=prefer_provider,
            cohere_model=cohere_model,
            groq_model=groq_model
        )
        for i in range(remaining)
    ]

    with open(output_path, "a", encoding="utf-8") as out_f:
        completed_count = existing_count
        for coro in tqdm_asyncio.as_completed(tasks, desc="Generating REE Dataset"):
            result = await coro
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                completed_count += 1

    print(f"\n✅ Generation complete! Total valid samples in {output_path}: {completed_count}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Async Synthetic Generator for REE Hydrometallurgy")
    parser.add_argument("--samples", type=int, default=5, help="Number of Q&A pairs to generate")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent async API calls")
    parser.add_argument("--provider", type=str, default="cohere", choices=["cohere", "groq"], help="Preferred provider")
    parser.add_argument("--cohere-model", type=str, default="command-r-plus-08-2024", help="Cohere model")
    parser.add_argument("--groq-model", type=str, default="llama-3.3-70b-versatile", help="Groq model")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="Output jsonl file")
    args = parser.parse_args()

    asyncio.run(
        run_batch_generation(
            total_samples=args.samples,
            concurrency=args.concurrency,
            output_path=Path(args.output),
            prefer_provider=args.provider,
            cohere_model=args.cohere_model,
            groq_model=args.groq_model
        )
    )


if __name__ == "__main__":
    main()
