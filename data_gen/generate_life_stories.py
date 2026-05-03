import os
import json
import argparse
import time
import threading
import concurrent.futures
from typing import List, Dict, Any
import httpx
from tqdm import tqdm

# MiMo API Configuration
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2-flash"

class StoryGenerator:
    def __init__(self, model_name=MIMO_MODEL):
        self.model_name = model_name
        self.api_key = os.getenv("MIMO_API_KEY")
        self.write_lock = threading.Lock() # Ensure thread-safe writing
        
        if not self.api_key:
            print("Warning: MIMO_API_KEY not found in environment variables.")
            print("Please set it: export MIMO_API_KEY='your_key'")
        
        # Configure httpx client settings
        self.client_settings = {
            "timeout": 120.0,
            "limits": httpx.Limits(max_keepalive_connections=20, max_connections=50)
        }

    def _append_line_to_file(self, data, filepath):
        """Appends a single JSON object as a new line to the file (Thread-Safe)."""
        with self.write_lock:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _call_llm(self, prompt: str) -> str:
        """Call MiMo API using httpx (Single attempt)."""
        if not self.api_key:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a master storyteller specializing in realistic, slice-of-life fiction. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_completion_tokens": 8192,
            "temperature": 0.8,
            "top_p": 0.95,
            "stream": False,
            "thinking": {"type": "disabled"}
        }
        
        with httpx.Client(**self.client_settings) as client:
            response = client.post(
                f"{MIMO_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    def generate_story_for_persona(self, persona: Dict[str, Any], max_retries=3) -> Dict[str, Any]:
        """
        Generates detailed life events based on the persona's 2020-2025 evolution.
        Includes robust retry logic for API and JSON validation.
        """
        name = persona.get("name", "Unknown")
        timeline = persona.get("evolution_timeline", [])
        
        # Format the psychological timeline for the prompt
        timeline_str = ""
        for phase in timeline:
            timeline_str += f"""
- Period: {phase.get('period')}
  - Mindset: {phase.get('internal_state')}
  - Goal: {phase.get('primary_goal')}
  - Shift: {phase.get('personality_shift')}
"""

        prompt = f"""
You are writing a detailed, structured life chronicle for a character named **{name}**.
I will provide you with their **Psychological Evolution Timeline** from 2020 to 2025.
Your task is to generate a dense sequence of **5 to 7 specific life events per year** based on their psychology.

Character Profile:
- Name: {name}
- Job: {persona.get('demographics', {}).get('occupation')}
- Location: {persona.get('demographics', {}).get('location')}
- Personality: {persona.get('current_profile', {}).get('personality_summary')}

Psychological Timeline (The "Why"):
{timeline_str}

Task:
Generate a **Year-by-Year Chronicle** (2020-2025).
For **EACH YEAR**, generate **5 to 7 separate events**.

Constraints & Guidelines:
1.  **Strict Causality**: You MUST track the ripple effects of choices using `caused_by_event_ids`. Small events (e.g., meeting a stranger) should cause big events later (e.g., getting a job offer from them).
    - **Long-term Causality**: Do not limit causality to the same year. An event in 2023 can and should be caused by something in 2020 or 2021 if relevant.
2.  **Temporal Spacing**: Events must be distributed throughout the year (e.g., Winter, Spring, Summer, Fall). Don't clump them all in January.
3.  **Topic Diversity**: Do NOT just write about their job. You must include events from at least 3 distinct categories per year:
    * *Career/Academic* (Work projects, promotions, failures)
    * *Social/Relationships* (Dating, friendships, conflicts)
    * *Personal/Hobby* (Learning new skills, solitary moments, health)
    * *Life Admin* (Moving houses, buying big items, financial decisions)
4.  **Show the Psychology**: Every event description must be an ACTION. The *internal* feeling belongs in the `psychological_note` field.
    - Bad: "He felt sad about the breakup."
    - Good: "He deleted all photos of his ex from his phone and booked a solo flight to Tokyo."

Output Format (JSON only):
{{
    "character_name": "{name}",
    "chronology": [
        {{
            "year": 2020,
            "narrative_summary": "A cohesive paragraph summarizing the year's arc.",
            "events": [
                {{
                    "event_id": "2020_E1",
                    "time_of_year": "February (Winter)",
                    "category": "Career",
                    "description": "Describe the event vividly. Include names, places, and specific actions.",
                    "psychological_note": "Briefly explain how this reflects their internal state (e.g., 'Fueled by imposter syndrome').",
                    "caused_by_event_ids": [] 
                }},
                {{
                    "event_id": "2020_E2",
                    "time_of_year": "May (Spring)",
                    "category": "Social",
                    "description": "Event triggered by the previous one...",
                    "psychological_note": "...",
                    "caused_by_event_ids": ["2020_E1"]
                }},
                ... (Generate 5 to 7 events)
            ]
        }},
        ... (Repeat for 2021, 2022, 2023, 2024, 2025)
    ]
}}
"""
        # RETRY LOOP: Handles API failures AND JSON Parse failures
        for attempt in range(max_retries):
            try:
                response_text = self._call_llm(prompt)
                
                if not response_text:
                    raise ValueError("Empty response from API")

                # Clean up response
                cleaned = response_text.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                story_data = json.loads(cleaned)
                return story_data
                
            except (json.JSONDecodeError, ValueError, Exception) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    # print(f"Retry {name} (Attempt {attempt+1}): {e}") 
                    time.sleep(wait_time)
                else:
                    print(f"Failed to generate story for {name} after {max_retries} attempts. Error: {e}")
                    return None
        return None

    def _load_jsonl(self, filepath):
        data = []
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
        return data

    def _append_line_to_file(self, data, filepath):
        """Appends a single JSON object as a new line to the file."""
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def process_batch(self, input_file, output_file, max_workers=10):
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            return

        # Load input personas (Standard JSON array)
        with open(input_file, 'r', encoding='utf-8') as f:
            personas = json.load(f)

        print(f"Loaded {len(personas)} personas from {input_file}")

        # Load existing progress (JSONL format)
        existing_stories = self._load_jsonl(output_file)
        completed_names = {s.get("character_name") for s in existing_stories}
        print(f"Loaded {len(existing_stories)} existing stories. Resuming...")
        
        personas_to_process = [p for p in personas if p.get("name") not in completed_names]
        print(f"Remaining to process: {len(personas_to_process)}")
        
        # Parallel Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_persona = {executor.submit(self.generate_story_for_persona, p): p for p in personas_to_process}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_persona), total=len(personas_to_process), desc="Generating Stories"):
                try:
                    story = future.result()
                    original_persona = future_to_persona[future]
                    
                    if story:
                        # Merge original persona data
                        story["original_persona"] = original_persona
                        
                        # Save IMMEDIATELY using append mode (JSONL)
                        # This fixes the Disk Quota Exceeded error by avoiding full rewrites
                        self._append_line_to_file(story, output_file)
                        
                except Exception as e:
                    print(f"Thread error: {e}")
        
        print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Generate Character Stories using MiMo")
    parser.add_argument("--input", type=str, required=True, help="Input raw personas JSON")
    parser.add_argument("--output", type=str, default="character_stories.jsonl", help="Output stories JSONL file")
    parser.add_argument("--model", type=str, default=MIMO_MODEL)
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel worker threads")
    
    args = parser.parse_args()
    
    generator = StoryGenerator(model_name=args.model)
    generator.process_batch(args.input, args.output, max_workers=args.workers)

if __name__ == "__main__":
    main()
