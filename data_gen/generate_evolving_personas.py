import os
import json
import time
import argparse
import random
import concurrent.futures
from typing import List, Dict, Any
import httpx
from tqdm import tqdm

# MiMo API Configuration
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2-flash"

class EvolvingPersonaGenerator:
    def __init__(self, model_name=MIMO_MODEL, bank_path=None):
        self.model_name = model_name
        self.api_key = os.getenv("MIMO_API_KEY")
        if not self.api_key:
            print("Warning: MIMO_API_KEY not found in environment variables.")
            print("Please set it: export MIMO_API_KEY='your_key'")
        self.bank_personas = self._load_bank(bank_path) if bank_path else []

    def _load_bank(self, path):
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Loaded {len(data)} personas from bank: {path}")
                    return data
            except Exception as e:
                print(f"Error loading bank: {e}")
        elif path:
            print(f"Warning: Bank file not found at {path}")
        return []

    def _call_llm(self, prompt: str) -> str:
        """Call MiMo API using httpx (Single attempt). error handling handled by caller."""
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
                    "content": "You are an expert creative writer and psychologist. Always respond with valid JSON only, no additional text or markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_completion_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False,
            "thinking": {"type": "disabled"}
        }
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{MIMO_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    def generate_evolution_from_base(self, base_persona: Dict[str, Any], max_retries=3) -> Dict[str, Any]:
        """
        Takes a base persona from the bank and generates the personality evolution 
        specifically for the years 2020 to 2025.
        Includes robust Retry logic for both API errors and JSON parsing errors.
        """
        # Extract fields from the bank format
        profile = base_persona.get("profile", {})
        name = profile.get("name", "Unknown")
        age = profile.get("age", "30")
        job = profile.get("job", "Unknown")
        mbti = base_persona.get("mbti", "Unknown")
        personality = profile.get("personality", "")
        growth_exp = profile.get("growth_experience", "")
        family = profile.get("family_relationship", "")
        worry = profile.get("recent_worry_or_anxiety", "")
        hobby = profile.get("hobby", "")
        tone = profile.get("tone", "")
        gender = profile.get("gender", "unknown")
        region = profile.get("region", "unknown")
        
        prompt = f"""
I will provide you with a detailed profile of a character named {name} as they are **Today (in 2025)**.
Your task is to **Reverse Engineer** their psychological journey from **2020 to 2025**.
You need to explain the internal evolution that led them to their current self.

Target Persona (State in 2025):
- Name: {name}
- Age: {age}
- Location: {region}
- MBTI: {mbti}
- Occupation: {job}
- Core Personality: {personality}
- Current Anxiety/Focus (2025): {worry} - *This is where they ended up.*
- History/Background: {growth_exp} - *Use this as deep backstory.*
- Family Dynamics: {family}

Task:
Construct a **Psychological Evolution Timeline** (2020-2025) that connects their past to this present state.
For each phase, describe their internal world. Focus on **Human Goals & Motivations** (e.g., career ambition, desire for connection, quest for meaning) and how those shifted.
This timeline will later be used to generate the specific life events they experienced, so focus on the **"Why"** (psychological drivers) rather than the "What" (specific events).

Output Format (JSON only, no markdown):
{{
    "name": "{name}",
    "demographics": {{ "age": {age}, "gender": "{gender}", "location": "{region}", "occupation": "{job}" }},
    "current_mbti": "{mbti}",
    "evolution_timeline": [
        {{
            "period": "2020-2021",
            "internal_state": "What was their primary mindset? (e.g., 'Seeking reinvention', 'Protective of loved ones')",
            "primary_goal": "What was their main driving objective? (e.g., 'Striving for financial stability', 'Focusing on self-improvement')",
            "personality_shift": "How did pursuing this goal shape their personality? (e.g., 'Became more disciplined', 'Developed more empathy')"
        }},
        {{
            "period": "2022-2023",
            "internal_state": "How did their mindset evolve?",
            "primary_goal": "What did they shift their focus towards? (e.g., 'Expanding social circle', 'Advancing career')",
            "personality_shift": "..."
        }},
        {{
            "period": "2024-2025",
            "internal_state": "Describe the psychological state that matches their 'Current Anxiety/Focus' in the profile.",
            "primary_goal": "What are they currently striving for?",
            "personality_shift": "..."
        }}
    ],
    "current_profile": {{
        "personality_summary": "A cohesive summary of their character in 2025, bridging their core MBTI with this recent history.",
        "tone_of_voice": "{tone}",
        "hobbies": "{hobby}",
        "recent_anxiety_or_goal": "{worry}"
    }}
}}
"""
        # RETRY LOOP: Handles both API failures AND JSON Parse failures
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
                
                # Parse JSON
                return json.loads(cleaned)
                
            except (json.JSONDecodeError, ValueError, Exception) as e:
                # Capture JSON errors, API errors, etc.
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    # Only print verbose errors if it's the last attempt or critical, otherwise keep logs cleaner
                    # print(f"Retry {name} (Attempt {attempt+1}): {e}") 
                    time.sleep(wait_time)
                else:
                    print(f"Failed to generate {name} after {max_retries} attempts. Error: {e}")
                    # print(f"Last Raw Response: {response_text[:200]}...") # Debug only
                    return None
        return None

    def generate_random_batch(self, count=5) -> List[Dict]:
        """
        Original method (fallback): Generates personas from scratch if no bank provided.
        """
        prompt = f"""
Generate {count} detailed fictional user personas.
These personas must demonstrate **Dynamic Personality Evolution**.

Output Format (JSON array only, no markdown):
[
    {{
        "name": "Full Name",
        "demographics": {{ "age": 30, "gender": "male/female", "location": "City, Country", "occupation": "Job Title" }},
        "current_mbti": "INTJ",
        "evolution_timeline": [
            {{
                "stage": "Childhood",
                "age_range": "0-12",
                "dominant_traits": ["curious", "introverted"],
                "key_event": "A formative event description",
                "impact_change": "How it changed the person"
            }}
        ],
        "current_profile": {{
            "personality_summary": "Description of current personality",
            "tone_of_voice": "calm, analytical",
            "hobbies": ["reading", "chess"],
            "recent_anxiety_or_goal": "Current concern or goal"
        }}
    }}
]
"""
        response_text = self._call_llm(prompt)
        if not response_text:
            return []
            
        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return []

    def generate_personas(self, count=5, max_workers=10) -> List[Dict]:
        results = []
        
        if self.bank_personas:
            # Select random entries from bank (Safe: happens before threading)
            # Ensure we don't exceed available unique personas if count > len
            num_unique_needed = min(count, len(self.bank_personas))
            selected_bases = random.sample(self.bank_personas, num_unique_needed)
            
            # If we need more than available unique ones, sample with replacement for the rest
            if count > len(self.bank_personas):
                extra = random.choices(self.bank_personas, k=count - len(self.bank_personas))
                selected_bases.extend(extra)
            
            print(f"Generating {len(selected_bases)} personas using Bank Data with {max_workers} threads...")
            
            # Parallel Execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_base = {executor.submit(self.generate_evolution_from_base, base): base for base in selected_bases}
                
                # Process as they complete
                for future in tqdm(concurrent.futures.as_completed(future_to_base), total=len(selected_bases), desc="Evolving Personas"):
                    try:
                        persona = future.result()
                        if persona:
                            results.append(persona)
                    except Exception as e:
                        print(f"Thread error: {e}")
                        
        else:
            print("No bank data. Generating from scratch...")
            results = self.generate_random_batch(count)
            
        return results

    def save_personas(self, personas, output_file):
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                    if isinstance(existing, list):
                        existing.extend(personas)
                        personas = existing
            except Exception:
                pass
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(personas, f, indent=4, ensure_ascii=False)
        print(f"Saved total {len(personas)} personas to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Evolving Personas using MiMo API")
    parser.add_argument("--count", type=int, default=5, help="Number of personas to generate")
    parser.add_argument("--output", type=str, default="evolving_personas.json", help="Output JSON file")
    parser.add_argument("--model", type=str, default=MIMO_MODEL, help="Model name")
    parser.add_argument("--bank_file", type=str, default=None, help="Path to MBTI Bank JSON")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel worker threads")
    
    args = parser.parse_args()
    
    generator = EvolvingPersonaGenerator(model_name=args.model, bank_path=args.bank_file)
    
    personas = generator.generate_personas(count=args.count, max_workers=args.workers)
    
    if personas:
        generator.save_personas(personas, args.output)
        print("Done.")
    else:
        print("Failed to generate personas.")

if __name__ == "__main__":
    main()
