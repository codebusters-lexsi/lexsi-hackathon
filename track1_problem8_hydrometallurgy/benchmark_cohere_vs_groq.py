"""
Benchmark script: Measure exact throughput, latency, and tokens/sec
comparing Cohere Labs vs Groq Cloud.
"""

import os
import time
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

TEST_PROMPT = """You are a Principal Hydrometallurgical Engineer.
Explain in detail the equilibrium reaction and McCabe-Thiele stage calculation
for separating Nd3+ and Pr3+ using 1.2 M PC-88A in kerosene at pH 3.2.
Provide exact equations and numerical parameters."""

SYSTEM_PROMPT = "You are an expert hydrometallurgist. Output strictly valid JSON with 'messages': [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]"

async def test_cohere(model="command-r-plus-08-2024", n=2):
    import cohere
    client = cohere.AsyncClientV2(api_key=os.getenv("COHERE_API_KEY"))
    start = time.perf_counter()
    tokens = 0
    for i in range(n):
        t0 = time.perf_counter()
        res = await client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": TEST_PROMPT}
            ],
            response_format={"type": "json_object"}
        )
        t_sample = time.perf_counter() - t0
        text = res.message.content[0].text
        t_est = len(text) // 4
        tokens += t_est
        print(f"[{model} Sample {i+1}] {t_sample:.2f}s | ~{t_est} tokens | {t_est/t_sample:.1f} tok/s")
    total_time = time.perf_counter() - start
    return {"provider": f"Cohere ({model})", "samples": n, "total_time": total_time, "avg_time_per_sample": total_time/n, "tokens": tokens, "avg_tok_per_sec": tokens/total_time}

async def test_groq(model="openai/gpt-oss-120b", n=2):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    start = time.perf_counter()
    tokens = 0
    for i in range(n):
        t0 = time.perf_counter()
        res = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": TEST_PROMPT}
            ],
            response_format={"type": "json_object"}
        )
        t_sample = time.perf_counter() - t0
        text = res.choices[0].message.content
        t_est = len(text) // 4
        tokens += t_est
        print(f"[{model} Sample {i+1}] {t_sample:.2f}s | ~{t_est} tokens | {t_est/t_sample:.1f} tok/s")
    total_time = time.perf_counter() - start
    return {"provider": f"Groq ({model})", "samples": n, "total_time": total_time, "avg_time_per_sample": total_time/n, "tokens": tokens, "avg_tok_per_sec": tokens/total_time}

async def main():
    print("=== 1. Testing Groq Cloud (openai/gpt-oss-120b) ===")
    g_res = await test_groq("openai/gpt-oss-120b", 2)
    
    print("\n=== 2. Testing Cohere Labs (command-r-plus-08-2024) ===")
    c_plus_res = await test_cohere("command-r-plus-08-2024", 2)

    print("\n=== 3. Testing Cohere Labs (command-r-08-2024) ===")
    c_r_res = await test_cohere("command-r-08-2024", 2)

    print("\n=== SUMMARY COMPARISON ===")
    for r in [g_res, c_plus_res, c_r_res]:
        print(f"{r['provider']}: Avg {r['avg_time_per_sample']:.2f}s/sample | Speed: {r['avg_tok_per_sec']:.1f} tok/s")

if __name__ == "__main__":
    asyncio.run(main())
