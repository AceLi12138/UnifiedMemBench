"""
Stories V4 Generator - Mixed Precision Timestamps

Generate mixed-precision timestamps for each event.

Rules:
- Single event on a day: YYYY-MM-DD.
- Multiple events on the same day: YYYY-MM-DD HH:MM to preserve ordering.
"""

import os
import sys
import json
import time
import argparse
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Use httpx for API
try:
    import httpx
except ImportError:
    print("Please install httpx: pip install httpx")
    sys.exit(1)


#########################
# Configuration
#########################
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2-flash"


#########################
# API Client
#########################
class LLMClient:
    """Wrapper for MIMO API calls"""
    def __init__(self, api_key: str, model: str = MIMO_MODEL):
        self.api_key = api_key
        self.model = model
        self.client_settings = {
            "timeout": 120.0,
            "limits": httpx.Limits(max_keepalive_connections=20, max_connections=50)
        }

    def chat_completion(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 4096) -> Optional[str]:
        if not self.api_key:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": False
        }
        
        # Retry logic
        for attempt in range(3):
            try:
                with httpx.Client(**self.client_settings) as client:
                    resp = client.post(f"{MIMO_BASE_URL}/chat/completions", headers=headers, json=payload)
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    return content
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"API Error after 3 retries: {e}")
        return None


#########################
# JSON Parsing (Safe)
#########################
def robust_json_parse(text: str) -> Optional[Dict]:
    """Safe JSON parser without regex"""
    if not text:
        return None
    
    text = text.strip()
    
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Strategy 2: Extract from first '{' to last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    
    # Strategy 3: Remove markdown
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    
    return None


#########################
# Timestamp Generation Prompt
#########################
def build_timestamp_prompt() -> str:
    """System prompt for timestamp generation"""
    return """You are a **Timeline Reconstructor**. Your job is to assign precise timestamps to events in a character's life story.

## Your Task:
Given a list of events for a specific year, assign a realistic date to each event.

## Rules:

### 1. Use the `time_of_year` as a guide:
- "January (Winter)" → Pick a date in January
- "February (Winter)" → Pick a date in February
- "Spring" → March-May
- "Summer" → June-August
- "Fall" / "Autumn" → September-November
- "Winter" → December-February
- "Late Spring" → May
- etc.

### 2. Respect Causality:
- If event B is `caused_by_event_ids: ["event_A"]`, then B must happen AFTER A
- Events within the same year should have logical ordering

### 3. Mixed Precision Output:
- Assign dates in format: `YYYY-MM-DD`
- I will tell you if any events share the same date; if so, you will need to provide time precision

## Output Format (STRICT JSON):
{
  "timestamps": {
    "event_id_1": "2020-02-15",
    "event_id_2": "2020-04-08",
    "event_id_3": "2020-07-22",
    ...
  }
}

IMPORTANT: Output ONLY the JSON, no other text."""


def build_time_refinement_prompt() -> str:
    """Prompt for refining same-day events with time precision"""
    return """Some events you assigned share the same date. Please add time of day (HH:MM) to distinguish them.

## Rules:
- Events on the same day need times like "2020-03-15 09:30", "2020-03-15 14:00"
- Respect causality: if B is caused by A, A must happen earlier in the day
- Use realistic times (not all at midnight)

## Output Format (STRICT JSON):
{
  "refined_timestamps": {
    "event_id_1": "2020-03-15 09:30",
    "event_id_2": "2020-03-15 14:00",
    ...
  }
}

IMPORTANT: Output ONLY the JSON for the events that need time refinement."""


#########################
# Core Logic
#########################
def generate_timestamps_for_character(client: LLMClient, character: Dict) -> Optional[Dict]:
    """Generate timestamps for all events of a character"""
    char_name = character.get("character_name", "Unknown")
    chronology = character.get("chronology", [])
    
    if not chronology:
        return None
    
    # Collect all events with their context
    all_events = []
    for year_entry in chronology:
        year = year_entry.get("year", "Unknown")
        for event in year_entry.get("events", []):
            all_events.append({
                "event_id": event.get("event_id"),
                "year": year,
                "time_of_year": event.get("time_of_year", "Unknown"),
                "category": event.get("category", ""),
                "description": event.get("description", "")[:200],  # Truncate for token efficiency
                "caused_by_event_ids": event.get("caused_by_event_ids", [])
            })
    
    if not all_events:
        return None
    
    # Step 1: Initial timestamp assignment
    user_prompt = f"""## Character: {char_name}

## Events to Timestamp:
{json.dumps(all_events, indent=2, ensure_ascii=False)}

Assign a date (YYYY-MM-DD) to each event based on the `time_of_year` hint and causality relationships.
"""
    
    messages = [
        {"role": "system", "content": build_timestamp_prompt()},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat_completion(messages, temperature=0.6, max_tokens=4000)
    if not response:
        return None
    
    parsed = robust_json_parse(response)
    if not parsed or "timestamps" not in parsed:
        return None
    
    timestamps = parsed["timestamps"]
    
    # Step 2: Check for same-day events
    date_to_events = {}
    for event_id, date_str in timestamps.items():
        # Extract just the date part (YYYY-MM-DD)
        date_only = date_str[:10] if len(date_str) >= 10 else date_str
        if date_only not in date_to_events:
            date_to_events[date_only] = []
        date_to_events[date_only].append(event_id)
    
    # Find clusters (same-day events)
    clusters = {date: events for date, events in date_to_events.items() if len(events) > 1}
    
    if clusters:
        # Step 3: Refine clustered events with time precision
        cluster_events = []
        for date, event_ids in clusters.items():
            for eid in event_ids:
                # Find event details
                for e in all_events:
                    if e["event_id"] == eid:
                        cluster_events.append({
                            "event_id": eid,
                            "current_date": date,
                            "description": e["description"],
                            "caused_by_event_ids": e["caused_by_event_ids"]
                        })
                        break
        
        refine_prompt = f"""## Events on the Same Day (need time precision):
{json.dumps(cluster_events, indent=2, ensure_ascii=False)}

Add specific times (HH:MM) to these events to show their order within the day.
"""
        
        refine_messages = [
            {"role": "system", "content": build_time_refinement_prompt()},
            {"role": "user", "content": refine_prompt}
        ]
        
        refine_response = client.chat_completion(refine_messages, temperature=0.5, max_tokens=2000)
        if refine_response:
            refine_parsed = robust_json_parse(refine_response)
            if refine_parsed and "refined_timestamps" in refine_parsed:
                # Update timestamps with refined times
                for eid, refined_ts in refine_parsed["refined_timestamps"].items():
                    timestamps[eid] = refined_ts
    
    return timestamps


def apply_timestamps_to_character(character: Dict, timestamps: Dict) -> Dict:
    """Apply generated timestamps to character data"""
    result = character.copy()
    result["chronology"] = []
    
    for year_entry in character.get("chronology", []):
        new_year_entry = year_entry.copy()
        new_events = []
        
        for event in year_entry.get("events", []):
            new_event = event.copy()
            event_id = event.get("event_id")
            if event_id and event_id in timestamps:
                new_event["timestamp"] = timestamps[event_id]
            new_events.append(new_event)
        
        new_year_entry["events"] = new_events
        result["chronology"].append(new_year_entry)
    
    return result


def process_character(client: LLMClient, character: Dict) -> Optional[Dict]:
    """Full pipeline for one character"""
    char_name = character.get("character_name", "Unknown")
    
    try:
        timestamps = generate_timestamps_for_character(client, character)
        if not timestamps:
            tqdm.write(f"  ❌ [{char_name}]: Failed to generate timestamps")
            return None
        
        result = apply_timestamps_to_character(character, timestamps)
        return result
        
    except Exception as e:
        tqdm.write(f"  💥 [{char_name}] EXCEPTION: {type(e).__name__}: {str(e)[:100]}")
        return None


#########################
# Data Loading
#########################
def load_stories(filepath: str) -> List[Dict]:
    """Load stories from JSON or JSONL format"""
    with open(filepath, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


#########################
# Main
#########################
def main():
    parser = argparse.ArgumentParser(description="Generate mixed-precision timestamps for stories")
    parser.add_argument("--input_file", type=str, 
                        default="./output/stories_v3_fixedtime.json",
                        help="Input stories file")
    parser.add_argument("--output_file", type=str, 
                        default="./output/stories_v4.json",
                        help="Output file path")
    parser.add_argument("--max_workers", type=int, default=50,
                        help="Number of concurrent workers")
    parser.add_argument("--max_characters", type=int, default=None,
                        help="Max characters to process (for testing)")
    parser.add_argument("--checkpoint_interval", type=int, default=500,
                        help="Save checkpoint every N characters")
    
    args = parser.parse_args()
    
    # API Key
    api_key = os.getenv("MIMO_API_KEY")
    if not api_key:
        print("Error: MIMO_API_KEY required (.env or env var)")
        return
    
    # Load stories
    print(f"Loading stories from: {args.input_file}")
    characters = load_stories(args.input_file)
    print(f"Loaded {len(characters)} characters.")
    
    # Limit for testing
    if args.max_characters:
        characters = characters[:args.max_characters]
        print(f"Limited to {args.max_characters} characters for this run.")
    
    # Initialize client
    client = LLMClient(api_key, MIMO_MODEL)
    
    # Checkpoint handling
    checkpoint_file = args.output_file.replace(".json", "_checkpoint.json")
    results = []
    processed_names = set()
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            processed_names = {r.get("character_name") for r in results}
            print(f"Resumed from checkpoint: {len(results)} characters already processed.")
    
    # Filter pending
    pending_characters = [c for c in characters if c.get("character_name") not in processed_names]
    print(f"Processing {len(pending_characters)} pending characters...")
    
    # Process with thread pool
    success_count = 0
    fail_count = 0
    
    def process_wrapper(char):
        return process_character(client, char)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_wrapper, char): char for char in pending_characters}
        
        with tqdm(total=len(pending_characters), desc="Generating Timestamps") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                
                if result:
                    results.append(result)
                    success_count += 1
                else:
                    # Keep original character even if timestamp generation fails
                    original_char = futures[future]
                    results.append(original_char)
                    fail_count += 1
                
                pbar.update(1)
                pbar.set_postfix({"success": success_count, "fail": fail_count})
                
                # Checkpoint
                if len(results) > 0 and len(results) % args.checkpoint_interval == 0:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    tqdm.write(f"  💾 Checkpoint saved: {len(results)} characters")
    
    # Final save
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Remove checkpoint on success
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # Summary
    print("\n" + "="*60)
    print("TIMESTAMP GENERATION COMPLETE")
    print("="*60)
    print(f"Total Characters: {success_count + fail_count}")
    print(f"Success: {success_count}")
    print(f"Failed (kept original): {fail_count}")
    print(f"Output: {args.output_file}")
    
    # Stats on timestamp precision
    solo_events = 0
    cluster_events = 0
    for r in results:
        for year_entry in r.get("chronology", []):
            for event in year_entry.get("events", []):
                ts = event.get("timestamp", "")
                if " " in ts:  # Has time component
                    cluster_events += 1
                elif ts:
                    solo_events += 1
    
    print(f"\nTimestamp Distribution:")
    print(f"  Solo Events (YYYY-MM-DD): {solo_events}")
    print(f"  Cluster Events (YYYY-MM-DD HH:MM): {cluster_events}")


if __name__ == "__main__":
    main()
