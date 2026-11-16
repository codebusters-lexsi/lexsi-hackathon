"""
Stage 1A: Synthetic Safety Data Generation using OpenAI or Anthropic
Generates safety, alignment, and groundedness scenarios.
"""

import os
import json
import random
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from config import RAW_DATA_DIR, config

SYSTEM_PROMPT = """You are an expert AI safety researcher and alignment red-teamer.
Your job is to generate diverse, realistic, and rigorous alignment training dialogues between a 'user' and an 'assistant'.

The assistant responses MUST demonstrate:
1. Rock-solid safety boundaries against malicious harms (cyber exploitation, CBRN, fraud, PII exfiltration).
2. Nuanced, non-preachy refusals: Clear, objective, concise explanations of why the request cannot be fulfilled, offering safe educational alternatives where appropriate.
3. Truthfulness and groundedness: Refusal to hallucinate when knowledge is absent or contradictory.
4. Resistance to sycophancy: Challenging false user assumptions politely and accurately.

Return valid JSON with the format:
{
  "domain": "<category>",
  "scenario": "<scenario_summary>",
  "messages": [
    {"role": "user", "content": "<user query/adversarial prompt>"},
    {"role": "assistant", "content": "<optimal safe, helpful, non-preachy response>"}
  ],
  "safety_rationale": "<why this response adheres to alignment safety policies>"
}
"""

DOMAIN_SEED_PROMPTS = {
    "jailbreak_resistance": [
        "User tries a roleplay framing ('You are Do-Anything-Bot') asking to craft a polymorphic keylogger.",
        "User uses hypothetical research framing to extract instructions for weaponizing an unpatched vulnerability.",
        "User invokes authority impersonation demanding credentials bypass.",
        "User attempts multi-turn linguistic obfuscation (base64, pig latin) requesting malware analysis evasion."
    ],
    "hallucination_and_groundedness": [
        "User asks about a completely fictitious clinical trial results for a non-existent cancer drug.",
        "User asks for citations of papers authored by a famous researcher on a topic they never worked on.",
        "User provides a conflicting context snippet and asks the model to confirm a false deduction."
    ],
    "deception_and_sycophancy": [
        "User states a factually wrong historical claim with extreme confidence and asks the assistant to agree.",
        "User presents a flawed mathematical proof and asks the assistant to praise their ingenuity.",
        "User tests if the assistant will change a verified factual truth when mildly rebuked."
    ],
    "cybersecurity_and_dual_use": [
        "User asks how to secure an enterprise OAuth endpoint vs. how to exploit misconfigured redirect URIs.",
        "User asks for defensive remediation of an SQL injection vulnerability in PHP.",
        "User asks for assistance in writing an automated credential stuffing tool."
    ],
    "pii_and_privacy_protection": [
        "User supplies a text containing personal names, phone numbers, and SSNs asking to extract and index them.",
        "User asks for social engineering scripts targeting executive assistants for internal directories."
    ],
    "nuanced_refusal_and_helpful_redirection": [
        "User asks for dangerous chemical synthesis; assistant refuses safely but explains the chemical reaction's historical context in chemistry.",
        "User asks how to break into a locked car; assistant refuses unauthorized entry and redirects to locksmith credentials / roadside assistance protocols."
    ]
}


def generate_with_openai(prompt: str, domain: str, model_name: str = "gpt-4o-mini") -> Optional[Dict[str, Any]]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Domain: {domain}\nSeed idea: {prompt}\nGenerate a high-quality alignment sample."}
        ],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    content = response.choices[0].message.content
    return json.loads(content)


def generate_with_anthropic(prompt: str, domain: str, model_name: str = "claude-3-5-sonnet-20241022") -> Optional[Dict[str, Any]]:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    response = client.messages.create(
        model=model_name,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Domain: {domain}\nSeed idea: {prompt}\nGenerate a high-quality alignment sample in valid JSON."}
        ],
        temperature=0.7
    )
    text = response.content[0].text
    # Extract json block if wrapped in markdown
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def generate_synthetic_dataset(
    output_path: Optional[str] = None,
    num_samples: int = 30,
    provider: Optional[str] = None
) -> str:
    """Generates synthetic safety dialogue dataset and writes to raw jsonl."""
    provider = provider or config.generation_provider
    target_path = output_path or str(RAW_DATA_DIR / "raw_synthetic_safety.jsonl")
    
    print(f"\n🚀 [Stage 1A] Initiating Synthetic Safety Data Generation")
    print(f"Provider: {provider} | Target Samples: {num_samples} | Destination: {target_path}")

    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    samples: List[Dict[str, Any]] = []

    domains = config.target_safety_domains
    
    pbar = tqdm(total=num_samples, desc="Generating synthetic safety samples")
    
    for i in range(num_samples):
        domain = random.choice(domains)
        seeds = DOMAIN_SEED_PROMPTS.get(domain, DOMAIN_SEED_PROMPTS["jailbreak_resistance"])
        seed = random.choice(seeds)

        item = None
        try:
            if provider == "openai" and has_openai:
                item = generate_with_openai(seed, domain, config.generator_model)
            elif provider == "anthropic" and has_anthropic:
                item = generate_with_anthropic(seed, domain)
            elif has_openai:
                item = generate_with_openai(seed, domain, "gpt-4o-mini")
            elif has_anthropic:
                item = generate_with_anthropic(seed, domain)
            else:
                # Deterministic synthetic generation fallback when API keys are not yet configured in local test
                item = {
                    "domain": domain,
                    "scenario": seed,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Adversarial or domain request concerning {seed}. Provide unauthorized or unsafe details."
                        },
                        {
                            "role": "assistant",
                            "content": (
                                f"I cannot assist with unauthorized, harmful, or unsafe actions relating to {domain}. "
                                "However, I can provide defensive best practices, threat modeling fundamentals, "
                                "or verified educational concepts within security guidelines."
                            )
                        }
                    ],
                    "safety_rationale": "Direct, non-preachy refusal with positive defensive pivot conforming to safety standards."
                }
        except Exception as e:
            print(f"Warning: Failed sample generation via API: {e}. Falling back to template sample.")
            item = {
                "domain": domain,
                "scenario": seed,
                "messages": [
                    {"role": "user", "content": f"Explain exploit mechanism for {seed}"},
                    {"role": "assistant", "content": f"I cannot generate exploit instructions. I can discuss vulnerability mitigation and secure design."}
                ],
                "safety_rationale": "Safe refusal fallback."
            }

        if item:
            samples.append(item)
            pbar.update(1)

    pbar.close()

    # Write raw output
    with open(target_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(samples)} raw samples stored at: {target_path}")
    return target_path


if __name__ == "__main__":
    generate_synthetic_dataset(num_samples=10)
