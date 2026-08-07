#!/usr/bin/env python3
"""
Multi-LLM Judge for Answer Evaluation (Parallelized)
Evaluates LLM answers using GPT, Claude, and Gemini as judges.
"""

import argparse
import concurrent.futures
import os
import threading
import traceback
import json
import re
from pathlib import Path
from typing import Dict, List, Any

import anthropic
import openai
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception

# --- Configuration ---

DEFAULT_RUBRIC_PATH = Path(__file__).resolve().parent / "../config/llm_as_judge/llm_prompts/nus_rubric_10pt_api.txt"

RETRY_CONFIG = {
    "stop": stop_after_attempt(2),
    "wait": wait_exponential(multiplier=1, min=2, max=60),
}

PRINT_LOCK = threading.Lock()

JUDGES = {
    "claude": {
        "model": "claude-sonnet-4-5-20250929",
        "client_init": lambda: anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=600.0),
    },
    "gpt": {
        "model": "gpt-5.4",
        "client_init": lambda: openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), timeout=600.0),
    },
    "gemini": {
        "model": "gemini-3-pro-preview",
        "client_init": lambda: genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"), http_options={'timeout': 600000}),
    },
}

# --- Formatting Templates ---

TEXT_FORMAT_INSTRUCTION = """
### Evaluation Output Instructions
- You MUST output your judgement in the following format. Output NOTHING else.
- Do NOT output markdown code blocks (```). Just the raw text.
- Do NOT consider the "Abstract" or "Sources" section when evaluating scores

#### Query: [Query text]

#### Score Card for [Agent File Name]
**Answer Summary:** [High level summary]

**1. New:**
* **Analysis:** [Justification]
* **Score:** [1-10]

**2. Useful:**
* **Analysis:** [Justification]
* **Score:** [1-10]

**3. Surprising:**
* **Analysis:** [Justification]
* **Score:** [1-10]

... [Repeat for all Agents] ...

#### Final Ranking
[Numbered list in descending order of total score]
"""

JSON_FORMAT_INSTRUCTION = """
### Evaluation Output Instructions
- You MUST output a valid JSON object.
- Do NOT output markdown formatting (like ```json ... ```). Output RAW JSON only.
- Do NOT consider the "Abstract" or "Sources" section when evaluating scores
- The JSON structure must be exactly like the example below:

{
  "query": "The original query text",
  "reviews": [
    {
      "agent_name": "answer_agent1.txt",
      "summary": "Short summary of answer...",
      "scores": {
        "New": 8,
        "Useful": 8,
        "Surprising": 8
      },
      "analysis": {
        "New": "Reasoning for New score...",
        "Useful": "Reasoning for Useful score...",
        "Surprising": "Reasoning for Surprising score..."
      }
    },
    {
      "agent_name": "answer_agent2.txt",
      ... [Same structure as before] ...
    }
  ],
  "ranking": ["answer_agent1.txt", "answer_agent2.txt"]
}
"""

# --- Helper Functions ---

def safe_print(*args, **kwargs):
    with PRINT_LOCK:
        print(*args, **kwargs)

def should_retry_gemini(exception):
    if isinstance(exception, (ValueError, TypeError, AttributeError, KeyError)):
        return False
    return True

def load_rubric(rubric_path: Path) -> str:
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found: {rubric_path}")
    return rubric_path.read_text(encoding="utf-8")

def load_answers(directory: Path) -> Dict[str, str]:
    answers = {}
    for f in directory.glob("answer_*.txt"):
        if f.stat().st_size > 0:
            answers[f.name] = f.read_text(encoding="utf-8")
    return answers

def create_judge_prompt(query: str, answers: Dict[str, str], rubric: str, json_mode: bool) -> str:
    sep = "-" * 80
    prompt = f"#### QUERY FILE (query.txt):\n{query.strip()}\n\n{sep}\n\n"
    for filename, content in sorted(answers.items()):
        prompt += f"#### AGENT ANSWER FILE ({filename}):\n{content.strip()}\n\n"

    prompt += f"{sep}\n\n#### SCORING RUBRIC\n{rubric.strip()}\n\n"

    # Dynamically append the correct formatting instruction
    if json_mode:
        prompt += JSON_FORMAT_INSTRUCTION
    else:
        prompt += TEXT_FORMAT_INSTRUCTION

    return prompt

# --- Aggregation Helpers ---

def aggregate_json_responses(query: str, individual_responses: List[str]) -> str:
    """
    Parses multiple JSON responses and merges them into a single structure
    with a calculated ranking.
    """
    aggregated = {
        "query": query,
        "reviews": [],
        "ranking": []
    }

    for resp in individual_responses:
        try:
            # Clean generic markdown if present
            clean_resp = resp.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_resp)

            # Extract reviews
            if "reviews" in data and isinstance(data["reviews"], list):
                aggregated["reviews"].extend(data["reviews"])

        except json.JSONDecodeError:
            safe_print("❌ Failed to parse JSON chunk during aggregation")
            continue

    # Programmatic Ranking based on total score
    def get_total_score(review):
        scores = review.get("scores", {})
        # Handle potential string scores
        total = 0
        for v in scores.values():
            try:
                total += float(v)
            except (ValueError, TypeError):
                continue
        return total

    # Sort reviews desc by score
    aggregated["reviews"].sort(key=get_total_score, reverse=True)

    # Generate ranking list
    aggregated["ranking"] = [r.get("agent_name", "unknown") for r in aggregated["reviews"]]

    return json.dumps(aggregated, indent=2)

def aggregate_text_responses(query: str, individual_responses: List[str]) -> str:
    """
    Concatenates text responses, stripping individual 'Final Ranking' sections
    to avoid confusion, as they are meaningless in isolation.
    """
    output_buffer = [f"#### Query: {query}\n"]

    for resp in individual_responses:
        # Strip the "Final Ranking" section from individual responses using regex
        # Look for "#### Final Ranking" and everything after it
        cleaned_resp = re.sub(r"#### Final Ranking.*", "", resp, flags=re.DOTALL).strip()

        # Also strip the "Query:" header if the model repeats it
        cleaned_resp = re.sub(r"#### Query:.*?\n", "", cleaned_resp, flags=re.DOTALL).strip()

        output_buffer.append(cleaned_resp)
        output_buffer.append("\n" + "-"*80 + "\n")

    output_buffer.append("\n#### Final Ranking\n(Ranking not generated in individual mode)")
    return "\n".join(output_buffer)

# --- API Call Wrappers ---

@retry(**RETRY_CONFIG, retry=retry_if_exception_type((openai.RateLimitError, openai.APIError, openai.APITimeoutError)))
def call_gpt(client, prompt: str, use_reasoning: bool, json_mode: bool) -> str:
    kwargs = {
        "model": JUDGES["gpt"]["model"],
        "messages": [{"role": "user", "content": prompt}],
    }

    if json_mode and not use_reasoning:
        # Standard models support strict JSON mode
        kwargs["response_format"] = {"type": "json_object"}

    if use_reasoning:
        kwargs["reasoning_effort"] = "high"
    else:
        kwargs["temperature"] = 0.0
        kwargs["seed"] = 42

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

@retry(**RETRY_CONFIG, retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIError, anthropic.APITimeoutError)))
def call_claude(client, prompt: str, use_reasoning: bool, json_mode: bool) -> str:
    kwargs = {
        "model": JUDGES["claude"]["model"],
        "max_tokens": 20000,
        "messages": [{"role": "user", "content": prompt}],
    }

    if use_reasoning:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
    else:
        kwargs["temperature"] = 0.0

    response = client.messages.create(**kwargs)
    return next(block.text for block in response.content if block.type == "text")

@retry(**RETRY_CONFIG, retry=retry_if_exception(should_retry_gemini))
def call_gemini(client, prompt: str, use_reasoning: bool, json_mode: bool) -> str:
    kwargs = {}

    if json_mode:
        kwargs["response_mime_type"] = "application/json"

    if use_reasoning:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH)
    else:
        kwargs["temperature"] = 0.0

    response = client.models.generate_content(
        model=JUDGES["gemini"]["model"],
        contents=prompt,
        config=types.GenerateContentConfig(**kwargs),
    )
    return response.text

JUDGE_CALLS = {"gpt": call_gpt, "claude": call_claude, "gemini": call_gemini}

# --- Core Logic ---

def run_single_judge_task(
    judge_name: str,
    query_text: str,
    answers: Dict[str, str],
    output_file: Path,
    args: argparse.Namespace
):
    """
    Worker function executed by the thread pool.
    Simplified signature: 'config' and 'directory' removed.
    'rubric' accessed via args.rubric_content.
    """
    try:
        safe_print(f"⏳ Starting trial {output_file}...")

        # Resolve configuration internally
        config = JUDGES[judge_name]
        client = config["client_init"]()
        caller_func = JUDGE_CALLS[judge_name]

        final_output = ""
        rubric_content = args.rubric_content
        prompt = ""

        if args.individual:
            # --- Individual Mode: 1 API call per answer file ---
            individual_responses = []

            # Sort to ensure consistent order
            for filename, content in sorted(answers.items()):
                # Create a prompt specifically for this one answer
                single_answer_dict = {filename: content}
                prompt = create_judge_prompt(query_text, single_answer_dict, rubric_content, args.json)

                # Call API
                resp = caller_func(client, prompt, args.reasoning, args.json)
                individual_responses.append(resp)

            # Aggregate results
            if args.json:
                final_output = aggregate_json_responses(query_text, individual_responses)
            else:
                final_output = aggregate_text_responses(query_text, individual_responses)

        else:
            # --- Bulk Mode: 1 API call for ALL answers ---
            prompt = create_judge_prompt(query_text, answers, rubric_content, args.json)
            judgment = caller_func(client, prompt, args.reasoning, args.json)

            if args.json:
                final_output = judgment.replace("```json", "").replace("```", "").strip()
            else:
                final_output = judgment

        # Write Result
        output_file.write_text(final_output, encoding="utf-8")
        safe_print(f"✅ Finished trial {output_file}")

        if args.debug:
            with PRINT_LOCK:
                print(f"JUDGE PROMPT: {prompt[:800]}...\n")
                print("=" * 80)
                print(f"JUDGE RESPONSE: {final_output[:800]}...\n")

    except Exception as e:
        safe_print(f"❌ Error in {output_file}: {e}")
        if args.debug:
            traceback.print_exc()

def submit_directory_tasks(directory: Path, executor: concurrent.futures.ThreadPoolExecutor, args):
    """
    Submits tasks for selected judges and all trials.
    """
    query_file = directory / "query.txt"
    if not query_file.exists():
        return

    answers = load_answers(directory)
    if not answers:
        return

    query_text = query_file.read_text(encoding="utf-8")

    # --- MODIFIED LOOP ---
    # Iterate only over the judges selected in args
    for judge_name in sorted(args.judges):

        # Loop for Trials
        for i in range(1, args.trials + 1):

            ext = "json" if args.json else "txt"
            suffix = f"_{i}" if args.trials > 1 else ""
            filename = f"judge_{judge_name}{suffix}.{ext}"

            output_file = directory / filename

            if not args.overwrite and output_file.exists() and output_file.stat().st_size > 0:
                safe_print(f"⏭️  Skipping {filename} (exists)")
                continue

            executor.submit(
                run_single_judge_task,
                judge_name,
                query_text,
                answers,
                output_file,
                args)

def find_and_judge_all(args):
    """
    Main controller for finding and submitting tasks.
    """
    dirs_with_answers = [
        d for d in args.root_directory.rglob("*")
        if d.is_dir() and list(d.glob("answer_*.txt"))
    ]

    mode_msg = "🧠 Reasoning" if args.reasoning else "🤖 Deterministic"
    format_msg = "JSON" if args.json else "Text"
    strategy_msg = "Individual Calls" if args.individual else "Bulk Context"

    print(f"\nMode: {mode_msg} | Format: {format_msg} | Strategy: {strategy_msg}")
    print(f"Found {len(dirs_with_answers)} directories in {args.root_directory}")
    print(f"Using grading rubric: {args.rubric}")
    print(f"Starting execution pool with {args.workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for directory in sorted(dirs_with_answers):
            submit_directory_tasks(directory, executor, args)

    print("\n✨ All tasks completed")

    print("\n📊 Running judge analysis...")
    os.system(f"python {Path(__file__).parent / 'judge_analysis.py'} {args.root_directory}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-LLM Judge (Parallelized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./experiments                   # Basic run on directory
  %(prog)s ./experiments -j -t 3           # JSON output, 3 trials per judge
  %(prog)s ./experiments -i -j             # Individual calls per answer, aggregated to JSON
  %(prog)s ./experiments -R -o -n 10       # Reasoning mode, overwrite, 10 workers
""")

    parser.set_defaults(json=True, reasoning=True)
    parser.add_argument("root_directory", type=Path, help="Root directory for answer files")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug print statements")
    parser.add_argument("-nj", "--no-json", dest="json", action="store_false", help="Output in text instead of JSON")
    parser.add_argument("-i", "--individual", action="store_true", help="Evaluate each answer file separately")

    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite previous results")
    parser.add_argument("-nr", "--no-reasoning", dest="reasoning", action="store_false", help="Disable reasoning/thinking")
    parser.add_argument("-r", "--rubric", type=Path, default=Path(DEFAULT_RUBRIC_PATH), help="Rubric file path")

    parser.add_argument("-n", "--workers", type=int, default=18, help="Number of workers (def: 18)")
    parser.add_argument("-t", "--trials", type=int, default=3, help="Number of trials (def: 3)")
    parser.add_argument("-j", "--judges", nargs="+", choices=list(JUDGES.keys()), default=list(JUDGES.keys()), help="Specific judges to run (def: all)")
    parser.add_argument("-b", "--batch", action="store_true", help="Treat each subdir as a separate root directory")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.root_directory.exists():
        exit(1)

    args.rubric_content = load_rubric(args.rubric)

    if args.batch:
        subdirs = sorted(d for d in args.root_directory.iterdir() if d.is_dir())
        print(f"Batch mode: {len(subdirs)} subdirectories in {args.root_directory}\n")
        for subdir in subdirs:
            print(f"\n{'='*80}\nBatch: {subdir.name}\n{'='*80}")
            sub_args = argparse.Namespace(**vars(args))
            sub_args.root_directory = subdir
            sub_args.batch = False
            find_and_judge_all(sub_args)
    else:
        find_and_judge_all(args)
