import argparse
import time
import json
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Real-life coding mode scenarios: Large context (codebase) + specific tasks
CODING_PROMPTS = [
    {
        "prefix": "You are an expert software engineer. Here is a large Python codebase for a distributed system using asyncio and gRPC...\n" + "import asyncio\n" * 1000 + "\nNow, implement a new feature for rate limiting using token bucket algorithm.",
        "max_tokens": 512
    },
    {
        "prefix": "Review the following Rust code for memory safety issues and performance bottlenecks:\n" + "fn main() { let x = 5; }\n" * 500 + "\nSuggest improvements.",
        "max_tokens": 1024
    },
    {
        "prefix": "Translate this Java Spring Boot project to Go using Gin framework. Maintain the same REST API structure.\n" + "@RestController public class Controller { ... }\n" * 200,
        "max_tokens": 2048
    }
]

def send_request(api_url, prompt_data):
    payload = {
        "model": "kimi-k2.5",
        "messages": [{"role": "user", "content": prompt_data["prefix"]}],
        "max_tokens": prompt_data["max_tokens"],
        "stream": True
    }
    
    start_time = time.time()
    ttft = None
    total_tokens = 0
    
    try:
        response = requests.post(api_url, json=payload, stream=True, timeout=300)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                if ttft is None:
                    ttft = time.time() - start_time
                
                # Simple token count estimation (or parse JSON)
                line_data = line.decode('utf-8')
                if line_data.startswith("data: "):
                    data_str = line_data[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                total_tokens += 1 # Approximation
                    except:
                        pass
        
        end_time = time.time()
        duration = end_time - start_time
        tps = total_tokens / (duration - ttft) if (duration - ttft) > 0 else 0
        
        return {
            "ttft": ttft,
            "tps": tps,
            "total_tokens": total_tokens,
            "duration": duration,
            "success": True
        }
    except Exception as e:
        print(f"Request failed: {e}")
        return {"success": False}

def run_benchmark(api_url, concurrency):
    print(f"Starting benchmark with concurrency={concurrency}...")
    results = []
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for _ in range(concurrency):
            prompt = np.random.choice(CODING_PROMPTS)
            futures.append(executor.submit(send_request, api_url, prompt))
        
        for future in futures:
            results.append(future.result())
    
    successful_results = [r for r in results if r["success"]]
    if not successful_results:
        print("No successful requests.")
        return

    avg_ttft = np.mean([r["ttft"] for r in successful_results])
    avg_tps = np.mean([r["tps"] for r in successful_results])
    total_tps = sum([r["tps"] for r in successful_results])
    
    print("\n--- Benchmark Metrics ---")
    print(f"Successful Requests: {len(successful_results)}/{concurrency}")
    print(f"Avg TTFT: {avg_ttft:.4f}s")
    print(f"Avg Throughput per request: {avg_tps:.2f} tokens/s")
    print(f"Total Cluster Throughput: {total_tps:.2f} tokens/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.get_default("api_url")
    parser.add_argument("--api-url", default="http://localhost:30001/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    
    run_benchmark(args.api_url, args.concurrency)
