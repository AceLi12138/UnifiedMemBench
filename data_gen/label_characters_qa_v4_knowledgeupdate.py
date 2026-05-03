"""
Character-Level QA Label Generation with Consistency Validation

Two-stage workflow:
Phase 1 - The Architect: generate expanded QA tasks from the full character timeline.
Phase 2 - The Editor: validate consistency and filter conflicting QA tasks.
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

TASK_TYPES = [
    "Information Extraction",
    "Multi-session Reasoning", 
    "Event Summarization",
    "Temporal Reasoning",
    "Knowledge Updating",
    "Memory Arbitration"
]

# Mapping from common variants to standard names
TASK_TYPE_ALIASES = {
    # Temporal Reasoning variants
    "timeline logic": "Temporal Reasoning",
    "timeline reasoning": "Temporal Reasoning",
    "time reasoning": "Temporal Reasoning",
    "temporal logic": "Temporal Reasoning",
    "temporal": "Temporal Reasoning",
    # Knowledge Updating variants
    "state tracking": "Knowledge Updating",
    "state update": "Knowledge Updating",
    "knowledge update": "Knowledge Updating",
    "attribute tracking": "Knowledge Updating",
    # Memory Arbitration variants
    "memory arbitration": "Memory Arbitration",
    "false premise": "Memory Arbitration",
    "loaded question": "Memory Arbitration",
    "premise correction": "Memory Arbitration",
    # Multi-session Reasoning variants
    "multi-session": "Multi-session Reasoning",
    "cross-event reasoning": "Multi-session Reasoning",
    "cross-event": "Multi-session Reasoning",
    # Event Summarization variants
    "summarization": "Event Summarization",
    "summary": "Event Summarization",
    # Information Extraction variants
    "information extraction": "Information Extraction",
    "fact extraction": "Information Extraction",
    "detail extraction": "Information Extraction",
}

def normalize_task_type(task_type: str) -> str:
    """Normalize task type to one of the 6 standard types."""
    if not task_type:
        return "Information Extraction"  # default
    
    # Exact match first
    if task_type in TASK_TYPES:
        return task_type
    
    # Check aliases (case-insensitive)
    lower = task_type.lower().strip()
    if lower in TASK_TYPE_ALIASES:
        return TASK_TYPE_ALIASES[lower]
    
    # Partial match
    for alias, standard in TASK_TYPE_ALIASES.items():
        if alias in lower or lower in alias:
            return standard
    
    # Fuzzy match to standard types
    for std_type in TASK_TYPES:
        if std_type.lower() in lower or lower in std_type.lower():
            return std_type
    
    return "Information Extraction"  # fallback


#########################
# API Client
#########################
class LLMClient:
    """Wrapper for MIMO API calls"""
    def __init__(self, api_key: str, model: str = MIMO_MODEL):
        self.api_key = api_key
        self.model = model
        self.client_settings = {
            "timeout": 180.0,  # 3 minutes timeout to prevent hanging
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
        
        # Retry logic with exponential backoff
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
# JSON Parsing (Safe - No Regex)
#########################
def robust_json_parse(text: str) -> Optional[Dict]:
    """
    Safe JSON parser without regex (avoids catastrophic backtracking).
    Uses simple string operations for reliability.
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Strategy 1: Direct parse (fastest path)
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Strategy 2: Extract from first '{' to last '}' (safe string slicing)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    
    # Strategy 3: Remove markdown code blocks (simple replace, no regex)
    cleaned = text
    for marker in ['```json', '```JSON', '```']:
        cleaned = cleaned.replace(marker, '')
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    
    return None



def extract_causal_chains(all_events: List[Dict]) -> List[Dict]:
    """
    Extract causal chains (A -> B -> C) from events.
    Returns: List of dicts describing chains.
    """
    # 1. Build Cause -> Effect map
    # event['caused_by_event_ids'] list causes for this event.
    # So if A is in B['caused_by'], then A -> B.
    
    adj_list = {} # Cause ID -> List of Effect IDs
    event_map = {e.get("event_id"): e for e in all_events if e.get("event_id")}
    
    for evt in all_events:
        effect_id = evt.get("event_id")
        if not effect_id: continue
        
        causes = evt.get("caused_by_event_ids", [])
        for cause_id in causes:
            if cause_id not in adj_list:
                adj_list[cause_id] = []
            adj_list[cause_id].append(effect_id)
            
    # 2. Find chains (DFS)
    chains = []
    
    def dfs(current_id, path):
        # path is list of event_ids
        # If path length >= 3, valid chain
        if len(path) >= 3:
            chains.append(list(path))
            
        # Stop if too long or no children
        if len(path) >= 5: 
            return

        for child in adj_list.get(current_id, []):
            # Avoid cycles
            if child not in path:
                dfs(child, path + [child])
                
    # Start DFS from all nodes
    for eid in event_map.keys():
        dfs(eid, [eid])
        
    # 3. Filter and Format
    # Sort by length descending, and keep unique paths
    # (Simple deduping logic: if A->B->C exists, A->B is redundant if we want max length)
    chains.sort(key=len, reverse=True)
    unique_chains = []
    seen_strs = set()
    
    for ch in chains:
        ch_str = ",".join(ch)
        # Check if this chain is a subcheck of an existing one?
        # Actually simpler: just take Top 5 Longest Distinct ones
        if ch_str not in seen_strs:
            unique_chains.append(ch)
            seen_strs.add(ch_str)
            
    selected_chains = unique_chains[:5]
    
    formatted = []
    for ch in selected_chains:
        # Build description
        chain_desc = []
        for eid in ch:
            evt = event_map.get(eid)
            if not evt: continue
            ts = evt.get("timestamp", "N/A")
            desc = evt.get("description", "")[:50].replace("\n", " ") + "..."
            chain_desc.append(f"[{eid} ({ts}) {desc}]")
        
        formatted.append({
            "ids": ch,
            "text": " -> ".join(chain_desc)
        })
        
    return formatted


#########################
# Phase 1: The Architect
#########################
def build_architect_prompt() -> str:
    """System prompt for Phase 1: Batch QA Generation with Anti-Leakage & Precise Timestamps"""
    return """You are a **Character Biography Architect**. Your job is to read ALL events in a character's life and create a comprehensive, coherent QA knowledge base.

## Your Source of Truth:
You possess a dataset with precise timestamps (YYYY-MM-DD HH:MM). **You MUST use these timestamps as the sole source of truth for all time-related answers.** 
- Ignore the 'year' field if it appears to conflict with the timestamp.
- Use explicit dates in your reasoning.

## Your Tasks:
1. **Read the entire character timeline** (all events across years)
2. **Expand with creative, plausible details** where needed (times, locations, colors, names, psychological states)
3. **Generate QA pairs for 6 task types** - at least 3 per type (18+ total) for filtering later
4. **Enable cross-event reasoning** - some questions should require info from MULTIPLE events

═══════════════════════════════════════════════════════════════
## 🚨 UNIVERSAL ANTI-LEAKAGE LAW (MUST OBEY) 🚨
═══════════════════════════════════════════════════════════════

**DEFINITION OF LEAKAGE**: When a Query contains information that directly reveals or enables trivial deduction of the Gold Answer.

### FORBIDDEN PATTERNS:
1. **Date/Number Leakage in Temporal Reasoning**
   - ❌ BAD: "He graduated in 2015 and started working in 2018. How many years between?"
   - ✅ GOOD: "How many years passed between his graduation and first job?"
   
2. **Entity Leakage in Information Extraction**
   - ❌ BAD: "What color is his red Toyota car?"
   - ✅ GOOD: "What color is his car?"
   
3. **Answer Echo in Multi-session Reasoning**
   - ❌ BAD: "How did moving to Seattle relate to his career in Seattle?"
   - ✅ GOOD: "How did his relocation relate to his career change?"

### ENFORCEMENT:
When drafting ANY Query, ask yourself: "Does this question contain words/numbers that ARE the answer?" If YES → REWRITE.

═══════════════════════════════════════════════════════════════

## Task Types & Generation Rules:

### 1. Information Extraction
- Questions about specific factual details
- Add concrete details in expansion (names, colors, quantities)
- Example: "What was the name of the hospital where he was treated?"

### 2. Multi-session Reasoning (Causal Chains)
- **Constraint**: Create a question that connects the start of a causal chain to the end.
- Ask how the final event was influenced by the initial event, or trace the chain reaction.
- **Explicit Dates**: When tracing cause and effect, verify that the cause (Event A) strictly precedes the effect (Event B) in time.
- Example: "How did the initial [Event A] eventually lead to [Event C]?"
- **Answer**: Must explain the chain reaction, referencing intermediate events (the bridge) and their timestamps.

### 3. Event Summarization  
- High-level questions about periods or themes
- **Specific Time Spans**: Summaries should cite precise ranges.
- Example: "Summarize his psychological state from March 2020 to January 2021." (Instead of "during 2020")

### 4. Temporal Reasoning (STRICT COMPUTATION)
- **Precise Calculation**: You must calculate the exact duration based on timestamps.
- **No Approximations**: "About a year" is unacceptable. "365 days" is required.
- **Good Answer**: "365 days passed (from 2020-03-14 to 2021-03-14)."
- **Micro-Temporal**: If events happen on the same day (HH:MM), ask about sequence ("did X happen before Y?").

### 5. Knowledge Updating (CURRENT STATE ONLY)
- **Constraint**: The query must ask about the CHARACTER'S CURRENT STATE (at the end of the timeline) regarding a changed attribute.
- **FORBIDDEN**: Do NOT ask "how did it change" or mention the old state in the question.
- **Good Query**: "As of late 2025, what is Lloyd's primary job?" or "Where does he currently live?"
- **Answer**: Must reflect ONLY the new/updated information. The explanation can mention the old state context, but the core answer is the new state.
- **Tracking Fields**: For this task type ONLY, you MUST include:
  - "old_state_value": The previous value of the attribute (before the update).
  - "old_state_event_id": The event ID where the previous state was established.

### 6. Memory Arbitration
- Create LOADED QUESTIONS with false premises built-in
- The query CONTAINS a false statement that must be corrected
- Example: "Since you bought the car in 2023, how..." (Actually 2021)
- Gold Answer MUST correct the false premise first

## CRITICAL RULES:
1. Generate **at least 3 QA per task type** (18+ total)
2. `answer_components` is MANDATORY - break answers into atomic facts
3. `source_event_ids` should list which event(s) the QA draws from
4. **NO ANSWER LEAKAGE** - Query must not contain answer entities or calculation factors
5. Keep expansions CONSISTENT across all QAs
6. Use EXACTLY these task_type names: "Information Extraction", "Multi-session Reasoning", "Event Summarization", "Temporal Reasoning", "Knowledge Updating", "Memory Arbitration"

## Output Format (STRICT JSON):
{
  "scenario_expansions": "Summary of all creative details you added (unified settings)",
  "candidate_tasks": [
    {
      "task_type": "Task Name (use exact names from the 6 types above)",
      "source_event_ids": ["event_id_1", "event_id_2"],
      "query": "Question text (NO LEAKAGE)",
      "gold_answer": "Complete answer (with specific dates if relevant)",
      "answer_components": ["Fact 1", "Fact 2", "..."],
      "old_state_value": "Optional (Required for Knowledge Updating)",
      "old_state_event_id": "Optional (Required for Knowledge Updating)"
    },
    ... (18+ entries covering all 6 types)
  ]
}
"""


def phase1_generate_qa(client: LLMClient, character: Dict) -> tuple[Optional[Dict], str]:
    """
    Phase 1: Generate candidate QA pairs for a character
    Returns: (result_dict, error_message)
    """
    char_name = character.get("character_name", "Unknown")
    chronology = character.get("chronology", [])
    
    # Build event summary (safely sort by year, handling str/int mix)
    events_text = []
    event_ids = []
    
    # Flatten all events to process timestamps
    all_events = []
    for year_entry in chronology:
        for event in year_entry.get("events", []):
            # [CRITICAL] Remove 'year' field to avoid dirty data issues
            # We strictly rely on 'timestamp' now.
            event.pop("year", None)
            all_events.append(event)
            
    # Sort by timestamp only
    def event_sort_key(e):
        return e.get("timestamp", "9999-12-31")
        
    all_events.sort(key=event_sort_key)
    
    for i, event in enumerate(all_events):
        event_id = event.get("event_id", f"unknown_{i}")
        event_ids.append(event_id)
        
        # Calculate time delta with next event if both have timestamps
        time_context = ""
        if i < len(all_events) - 1:
            next_event = all_events[i+1]
            ts1 = event.get("timestamp")
            ts2 = next_event.get("timestamp")
            if ts1 and ts2:
                try:
                    # Simple string comparison or partial parsing if needed
                    # ideally use datetime, but for prompt context, raw is often fine.
                    # Let's add explicit delta info if formats match
                    from datetime import datetime
                    fmt = "%Y-%m-%d %H:%M" if ":" in ts1 else "%Y-%m-%d"
                    # Try to parse
                    try: 
                        d1 = datetime.strptime(ts1, fmt)
                        d2 = datetime.strptime(ts2, fmt)
                        delta = d2 - d1
                        days = delta.days
                        seconds = delta.seconds
                        hours = seconds // 3600
                        time_context = f"  >> Time until next event: {days} days, {hours} hours"
                    except:
                        pass
                except:
                    pass

        events_text.append(f"""
[Event ID: {event_id}]
Timestamp: {event.get("timestamp", "N/A")} 
Category: {event.get("category", "N/A")}
Description: {event.get("description", "")}
Psychological Note: {event.get("psychological_note", "")}{time_context}
""")
    
    if not events_text:
        return None, "No events found in chronology"
    
    # [NEW] Extract Causal Chains for Multi-session Reasoning
    causal_chains = extract_causal_chains(all_events)
    causal_context = ""
    if causal_chains:
        causal_context = "\n## Detected Causal Chains (Use these for Multi-session Reasoning tasks):\n"
        for i, chain in enumerate(causal_chains):
            causal_context += f"Chain {i+1}: {chain['text']}\n"
    
    user_prompt = f"""## Character: {char_name}

## Complete Life Timeline (Sorted by Timestamp):
{"".join(events_text)}

{causal_context}

---
Generate at least 18 QA pairs (3+ per task type) based on this character's complete life story.
**Focus heavily on the timestamps provided.**
For Multi-session Reasoning, prioritize using the "Detected Causal Chains" provided above.
for Knowledge Updating, ask ONLY about the CURRENT state.
Use the event IDs for source_event_ids field.
Output strict JSON only.
"""
    
    messages = [
        {"role": "system", "content": build_architect_prompt()},
        {"role": "user", "content": user_prompt}
    ]
    
    # Retry loop for better reliability
    for attempt in range(5):
        # Removed verbose API call log
        response = client.chat_completion(messages, temperature=0.8 + attempt * 0.1, max_tokens=16000)
        
        if not response:
            # Removed verbose empty response log
            if attempt == 0:
                continue  # Retry
            return None, "API returned empty response (after retries)"
        
        parsed = robust_json_parse(response)
        
        if not parsed:
            if attempt == 0:
                continue  # Retry
            preview = response[:300] if len(response) > 300 else response
            return None, f"JSON parse failed. Preview: {preview}"
        
        # Flexible key matching for candidate_tasks
        tasks_key = None
        possible_keys = ["candidate_tasks", "tasks", "qa_pairs", "questions", "generated_tasks", "expanded_tasks"]
        for key in possible_keys:
            if key in parsed and isinstance(parsed[key], list) and len(parsed[key]) > 0:
                tasks_key = key
                break
        
        if not tasks_key:
            if attempt == 0:
                continue  # Retry
            return None, f"No valid tasks found. Got keys: {list(parsed.keys())}"
        
        tasks = parsed[tasks_key]
        if len(tasks) == 0:
            if attempt == 0:
                continue  # Retry
            return None, "Tasks list is empty"
        
        # Success!
        return {
            "character_name": char_name,
            "event_ids": event_ids,
            "scenario_expansions": parsed.get("scenario_expansions", ""),
            "candidate_tasks": tasks
        }, ""
    
    return None, "All attempts failed"


#########################
# Phase 2: The Editor
#########################
def build_editor_prompt() -> str:
    """System prompt for Phase 2: QA Auditor & Leakage Judge"""
    return """You are a **QA Auditor & Leakage Judge**. You have TWO critical responsibilities:

═══════════════════════════════════════════════════════════════
## 🔍 PRIMARY MISSION: LEAKAGE DETECTION & REWRITING
═══════════════════════════════════════════════════════════════

Your FIRST and MOST IMPORTANT job is to scan EVERY query for **Answer Leakage**.

### What is Leakage?
When the Query contains words, numbers, or entities that ARE part of the Gold Answer.

### Detection Rules:

1. **Temporal Reasoning Leakage**
   - SCAN: Does the query contain specific dates/durations that act as calculation factors or the answer itself?
   - CHECK: Are those years needed to calculate the answer?
   - If YES → LEAKAGE DETECTED
   - ACTION: Rewrite query to remove the dates, use relative references instead
   - Example Fix: "He graduated in 2015 and started in 2018. How many years?" 
     → "How many years passed between his graduation and first job?"

2. **Information Extraction Leakage**
   - SCAN: Does the query contain the answer entity itself?
   - Example: "What color is his RED car?" (answer is "red" - LEAKED!)
   - ACTION: Rewrite to remove the answer: "What color is his car?"

3. **Memory Arbitration Check**
   - These SHOULD contain false information (that's the task design)
   - Just verify the Gold Answer properly CORRECTS the false premise

### CRITICAL: DO NOT DELETE - REWRITE!
If you find leakage, do NOT remove the QA. Instead, REWRITE the query to fix it.

═══════════════════════════════════════════════════════════════
## 🔧 SECONDARY MISSION: Quality Assurance
═══════════════════════════════════════════════════════════════

### Check 1: Conflict Detection
- For example, if QA_1 says "car is red" and QA_2 says "car is blue", DELETE the lower-quality one
- Keep the one that aligns best with the unified scenario settings

### Check 2: Redundancy Detection  
- If two QAs ask essentially the same question in different words, keep only the better one

### Check 3: Coverage Guarantee
- After filtering, ensure EACH task type has at least 2 QAs
- If a type has <2, GENERATE a new non-conflicting QA to fill the gap

### Check 4: Task Type Standardization
- Ensure all task_type values use EXACTLY these names:
  - "Information Extraction"
  - "Multi-session Reasoning"
  - "Event Summarization"
  - "Temporal Reasoning"
  - "Knowledge Updating"
  - "Memory Arbitration"

### Check 5: Knowledge Updating Schema
- For "Knowledge Updating" tasks, ensure `old_state_value` and `old_state_event_id` fields are preserved.
- If missing, infer them if possible, or mark for regeneration.

═══════════════════════════════════════════════════════════════
## Output Format (STRICT JSON):
{
  "consistent_scenario_settings": "The unified expansion settings after resolving conflicts",
  "conflict_log": [
    "Resolved: car color conflict (chose red)",
    "Rewritten: Temporal Q3 had year leakage",
    "Removed: Duplicate of Q5",
    "Normalized: 'Timeline Logic' → 'Temporal Reasoning'"
  ],
  "validated_tasks": [
    {
      "task_type": "Exact Standard Name",
      "source_event_ids": [...],
      "query": "Clean query with NO leakage",
      "gold_answer": "...",
      "answer_components": [...],
      "old_state_value": "...",
      "old_state_event_id": "..."
    },
    ... (at least 2 per task type = minimum 12 entries)
  ]
}
"""



def validate_temporal_logic(task: Dict, character_events_map: Dict[str, Dict]) -> tuple[bool, str]:
    """
    Hard validation for temporal consistency.
    Returns: (is_valid, reason_if_invalid)
    """
    task_type = task.get("task_type", "")
    event_ids = task.get("source_event_ids", [])
    
    if not event_ids:
        return True, ""  # No events to check

    # [NEW] Check for Empty Content
    query = task.get("query", "").strip()
    answer = task.get("gold_answer", "").strip()
    if not query or not answer:
        return False, "Empty query or gold_answer" 
        
    # [NEW] Check for Date Leakage in Temporal Reasoning
    # Forbidden: Queries containing Month names (e.g. "March 15") which implies precise date leakage.
    # Years (e.g. "2020") are acceptable context.
    if task_type == "Temporal Reasoning":
        import re
        # [NEW] Check for Numeric Date Leakage (YYYY-MM or YYYY-MM-DD)
        # Matches 4 digits, hyphen, 2 digits. Banning this bans both YYYY-MM and YYYY-MM-DD.
        if re.search(r'\d{4}-\d{2}', query):
             return False, f"Potential Date Leakage: Query contains numeric date format ({query})"

        # Match Month names (common leakage like 'March 15')
        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        for month in months:
            if month in query:
                 return False, f"Potential Date Leakage: Query contains month name '{month}'"
        
    # 1. Check if all events exist
    timestamps = []
    for eid in event_ids:
        evt = character_events_map.get(eid)
        if not evt:
            # Event ID hallucination is common
            return False, f"References non-existent event: {eid}"
        
        ts_str = evt.get("timestamp", "")
        if ts_str and ts_str != "N/A":
            timestamps.append(ts_str)
        else:
            # Timestamp missing. Without 'year' fallback, we just skip adding to timestamps.
            # If we end up with too few timestamps, checks below will safeguard.
            pass
    
    # 2. Chronological Order Check for Multi-session/Causal Reasoning
    # If the task implies causality (Multi-session), events usually listed in cause->effect order
    # (This assumes source_event_ids are ordered logically. If not, this check might be too strict,
    # but for generated data, enforcing source order often helps consistency.)
    if task_type == "Multi-session Reasoning" and len(timestamps) >= 2:
        # Check simple sorted order
        sorted_ts = sorted(timestamps)
        if timestamps != sorted_ts:
            # It's not strictly increasing. 
            pass

    # [NEW] Knowledge Updating Schema Check
    if task_type == "Knowledge Updating":
        if "old_state_value" not in task or "old_state_event_id" not in task:
             return False, "Missing mandatory fields: old_state_value / old_state_event_id"

    # 3. Temporal Reasoning Calculation Check
    # If Gold Answer contains a number (days, years), try to match with real delta.
    if task_type == "Temporal Reasoning" and len(timestamps) >= 2:
        import re
        from datetime import datetime
        
        # Try to parse timestamps
        try:
            dts = []
            valid_ts = True
            for ts in timestamps:
                if ":" in ts:
                    dts.append(datetime.strptime(ts, "%Y-%m-%d %H:%M"))
                elif "-" in ts: # %Y-%m-%d
                    dts.append(datetime.strptime(ts, "%Y-%m-%d"))
                else: 
                    valid_ts = False
            
            if valid_ts and len(dts) >= 2:
                # Calculate simple range (min to max)
                dts.sort()
                diff = dts[-1] - dts[0]
                days_diff = diff.days
                years_diff = days_diff / 365.25
                
                answer_text = task.get("gold_answer", "").lower()
                
                # Check for "X years"
                year_match = re.search(r'(\d+(\.\d+)?)\s*years?', answer_text)
                if year_match:
                    val = float(year_match.group(1))
                    # Allow 1 margin or 10% margin
                    if abs(val - years_diff) > 1.0 and abs(val - years_diff) / (years_diff + 0.1) > 0.1:
                         return False, f"Temporal Hucullination: Answer says {val} years, actual is {years_diff:.2f} years"

                # Check for "X days"
                day_match = re.search(r'(\d+)\s*days?', answer_text)
                if day_match:
                    val = int(day_match.group(1))
                    if abs(val - days_diff) > 5: # 5 days margin
                         return False, f"Temporal Hucullination: Answer says {val} days, actual is {days_diff} days"

        except Exception:
            pass # Skip calculation check if parsing fails

    return True, ""


def phase2_validate_qa(client: LLMClient, phase1_result: Dict, character: Dict) -> Optional[Dict]:
    """Phase 2: Validate and deduplicate QA pairs"""
    char_name = phase1_result.get("character_name", "Unknown")
    scenario = phase1_result.get("scenario_expansions", "")
    candidates = phase1_result.get("candidate_tasks", [])
    
    if len(candidates) < 6:
        # Not enough candidates to validate
        return None

    # [NEW] Pre-build event map for hard validation
    character_events_map = {}
    chronology = character.get("chronology", [])
    for year_entry in chronology:
        for event in year_entry.get("events", []):
            eid = event.get("event_id")
            if eid:
                character_events_map[eid] = event
    
    user_prompt = f"""## Character: {char_name}

## Unified Scenario Expansions (from generation phase):
{scenario}

## Candidate QA Pairs to Review ({len(candidates)} entries):
{json.dumps(candidates, indent=2, ensure_ascii=False)}

---
Review these QAs for conflicts and redundancy.
Ensure each of the 6 task types has at least 2 high-quality entries.
Output strict JSON only.
"""
    
    messages = [
        {"role": "system", "content": build_editor_prompt()},
        {"role": "user", "content": user_prompt}
    ]
    
    # Phase 2 API call
    response = client.chat_completion(messages, temperature=0.5, max_tokens=4000)
    
    if not response:
        return None
    
    # Parse response (using safe parser)
    parsed = robust_json_parse(response)
    
    # Define fallback/processing function
    def prepare_result(tasks_list, conflict_log=None):
        final_tasks = []
        rejected_log = conflict_log or []
        
        for task in tasks_list:
            task["task_type"] = normalize_task_type(task.get("task_type", ""))
            
            # [CRITICAL] Schema Cleanup: Ensure old_state fields ONLY appear for Knowledge Updating
            if task["task_type"] != "Knowledge Updating":
                task.pop("old_state_value", None)
                task.pop("old_state_event_id", None)
            
            # [NEW] Check Temporal Logic Hard Rules
            is_valid, reason = validate_temporal_logic(task, character_events_map)
            if is_valid:
                final_tasks.append(task)
            else:
                rejected_log.append(f"Hard filtered {task.get('task_type')}: {reason}")
                
        return {
            "character_name": char_name,
            "original_events_summary": phase1_result.get("event_ids", []),
            "consistent_scenario_settings": parsed.get("consistent_scenario_settings", scenario) if parsed else scenario,
            "conflict_log": rejected_log,
            "validated_tasks": final_tasks
        }
    
    if not parsed or "validated_tasks" not in parsed:
        # Fallback: return phase1 results with minimal processing
        return prepare_result(candidates[:12], ["LLM Validation Failed, using raw candidates"])
    
    # Normal Path
    return prepare_result(parsed.get("validated_tasks", []), parsed.get("conflict_log", []))


#########################
# Full Pipeline for One Character
#########################
def process_character(client: LLMClient, character: Dict) -> Optional[Dict]:
    """Full 2-phase pipeline for one character with error protection"""
    char_name = character.get("character_name", "Unknown")
    
    try:
        # Phase 1: Generate candidates
        tqdm.write(f"  ▶ [{char_name}] Phase 1...")
        phase1_result, error_msg = phase1_generate_qa(client, character)
        if not phase1_result:
            tqdm.write(f"  ❌ [{char_name}] Phase 1: {error_msg}")
            return None
        tqdm.write(f"  ✓ [{char_name}] Phase 1: {len(phase1_result.get('candidate_tasks', []))} candidates")
        
        # Phase 2: Validate and filter with [NEW] timestamp validation
        tqdm.write(f"  ▶ [{char_name}] Phase 2...")
        # Pass character for hard validation
        phase2_result = phase2_validate_qa(client, phase1_result, character)
        if not phase2_result:
            tqdm.write(f"  ❌ [{char_name}] Phase 2: Validation failed")
            return None
        
        num_validated = len(phase2_result.get("validated_tasks", []))
        tqdm.write(f"  ✓ [{char_name}] Phase 2: {num_validated} validated")
        
        return phase2_result
        
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
    parser = argparse.ArgumentParser(description="Character-level QA labeling with consistency validation")
    parser.add_argument("--input_file", type=str, 
                        default="./output/stories_v4.json",
                        help="Input stories file (JSON or JSONL)")
    parser.add_argument("--output_file", type=str, 
                        default=None,
                        help="Output file path")
    parser.add_argument("--max_workers", type=int, default=50,
                        help="Number of concurrent workers")
    parser.add_argument("--max_characters", type=int, default=None,
                        help="Max characters to process (for testing)")
    parser.add_argument("--checkpoint_interval", type=int, default=500,
                        help="Save checkpoint every N characters")
    
    args = parser.parse_args()
    
    if args.output_file is None:
        args.output_file = f"{args.input_file.replace('.json', '')}_characters_qa.json"

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
        
        with tqdm(total=len(pending_characters), desc="Processing Characters") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                
                if result:
                    results.append(result)
                    success_count += 1
                else:
                    fail_count += 1
                
                pbar.update(1)
                pbar.set_postfix({"success": success_count, "fail": fail_count})
                
                # Log every success with count
                if result:
                    tqdm.write(f"  ✅ [{success_count}] {result.get('character_name', '?')}: {len(result.get('validated_tasks', []))} tasks")
                
                # Checkpoint - save every N successful results
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
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total Characters: {success_count + fail_count}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output: {args.output_file}")
    
    # Task distribution
    task_counts = {}
    for r in results:
        for task in r.get("validated_tasks", []):
            tt = task.get("task_type", "Unknown")
            task_counts[tt] = task_counts.get(tt, 0) + 1
    
    print("\nTask Type Distribution:")
    for tt in TASK_TYPES:
        count = task_counts.get(tt, 0)
        print(f"  {tt}: {count}")
    
    total_tasks = sum(task_counts.values())
    print(f"\nTotal QA Pairs: {total_tasks}")
    print(f"Avg per Character: {total_tasks / max(1, success_count):.1f}")


if __name__ == "__main__":
    main()
