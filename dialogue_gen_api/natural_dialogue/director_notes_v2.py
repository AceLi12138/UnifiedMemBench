"""
Strategies E and F: improved director-note generator (Director Notes V2).

Core changes:
  1. Filter reasoning-only components using should_verify_component().
  2. Provide QA-query context for incomplete answer fragments.
  3. Preserve complete answer-component text without VERBATIM/CONTEXT splitting.
  4. Use positive instructions instead of negative "DO NOT list it robotically" wording.
  5. Optionally inject dialogue-dynamics events.
"""

import json
import random
from typing import List, Dict, Any, Optional


def should_verify_component(task_type: str, component: str) -> bool:
    """
    Decide whether this component should appear in the dialogue rather than
    being a reasoning result that should not be directly surfaced.

    This mirrors the logic in strict_eval_qa_coverage.py.
    """
    comp_lower = component.lower().strip()

    if task_type == "Temporal Reasoning":
        if comp_lower.startswith("calculation"):
            return False

    if task_type == "Memory Arbitration":
        if comp_lower.startswith("correction"):
            return False
        if comp_lower.startswith("fundamental difference"):
            return False

    return True


def generate_director_notes_v2(
    seg_type: str,
    anchors: List[Any],
    topic: str,
    char_name: str,
    dynamic_event: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate improved director notes for one segment.

    Args:
        seg_type: Segment type ("opening", "closing", "filler", "info_anchor", or "transition").
        anchors: InformationAnchor objects assigned to this segment.
        topic: Segment topic description.
        char_name: Character name.
        dynamic_event: Optional dialogue-dynamics dict with an "instruction" field.

    Returns:
        Director-note text.
    """
    notes = []

    if seg_type == "opening":
        notes.append(_build_opening_notes(char_name))
    elif seg_type == "closing":
        notes.append(_build_closing_notes(char_name))
    elif seg_type == "filler":
        notes.append(_build_filler_notes(topic, char_name))
    elif seg_type == "info_anchor":
        notes.append(_build_info_anchor_notes(anchors, char_name))
    else:
        notes.append(_build_transition_notes(char_name))

    if dynamic_event:
        instruction = dynamic_event.get("instruction", "")
        if instruction:
            notes.append(f"\n{instruction.replace('{char_name}', char_name)}\n")

    return "\n".join(notes)


def _build_opening_notes(char_name: str) -> str:
    return f"""[SEGMENT: OPENING]
This is the beginning of a long, natural conversation with {char_name}.
- Establish the setting and mood through sensory details (sounds, smells, light, temperature).
- The interlocutor should introduce themselves and share why they are here.
- {char_name} responds warmly but naturally, with genuine small talk.
- Let the conversation breathe. No rush into any specific topic.
- Target: 6-8 turns of relaxed, natural dialogue.
"""


def _build_closing_notes(char_name: str) -> str:
    return f"""[SEGMENT: CLOSING]
This is the end of the conversation.
- Wrap up naturally. Reference something specific from earlier in the conversation.
- Both parties express genuine appreciation — not generic pleasantries.
- Maybe hint at future meetings, a shared plan, or a lingering thought.
- Target: 4-6 turns of natural closing dialogue.
"""


def _build_filler_notes(topic: str, char_name: str) -> str:
    return f"""[SEGMENT: ORGANIC CONVERSATION]
Topic: {topic}
- This segment builds rapport and makes the conversation feel real.
- Encourage tangents, specific anecdotes, and sensory details.
- The interlocutor can share their own related experiences — this is a two-way conversation.
- {char_name} should speak in their authentic voice, not in polished prose.
- Include at least one moment of humor, surprise, or mild disagreement.
- Target: 8-12 turns of rich, grounded dialogue.
- Stay on this topic. Do not introduce any specific life events unless they arise naturally from the topic.
"""


def _build_info_anchor_notes(anchors: List[Any], char_name: str) -> str:
    lines = [
        f"[SEGMENT: INFORMATION ANCHOR]",
        f"This segment contains specific information that should emerge naturally in the conversation.",
        "",
        _INFO_ANCHOR_PREAMBLE.replace("{char_name}", char_name),
    ]

    anchor_idx = 0
    for anchor in anchors:
        if anchor.anchor_type == "setup":
            anchor_idx += 1
            lines.append(_build_setup_instruction(anchor_idx, anchor, char_name))
        else:
            task_type = anchor.task_data.get("task_type", "")
            if not should_verify_component(task_type, anchor.content):
                continue
            anchor_idx += 1
            lines.append(_build_fragment_instruction(anchor_idx, anchor, char_name))

    lines.append(f"""
- Target: 8-12 turns.
- Let information surface as part of genuine memories and stories, not as answers to an invisible quiz.
""")
    return "\n".join(lines)


_INFO_ANCHOR_PREAMBLE = """EMBEDDING GUIDELINES:

You MUST include every fact listed below in the dialogue. Each fact is essential — if you skip it, the dialogue fails.

HOW TO EMBED FACTS WELL:
- All dates, names, numbers, and key terms from a fact MUST appear in the dialogue exactly as written.
- Wrap each fact in a specific memory: describe where {char_name} was, what they saw or felt, what the moment was like.
- If the fact contains a date like 2020-03-15, use exactly that date. Using a different date makes the dialogue useless.

Example — given the fact: "Equipment purchased on 2020-03-15"

  GOOD (the fact is there, wrapped in a vivid memory):
  "It was mid-March, the 15th I think, 2020. I'd just gotten my tax refund and thought, you know what, I'm buying that silicone kit I've been eyeing. Clicked 'order' before I could talk myself out of it."

  BAD (wrong date — this breaks the data):
  "It was March 23, 2020, when I decided to invest in some equipment."

- The interlocutor can steer the conversation toward the right topic, but should not ask quiz-like questions.
"""


def _build_setup_instruction(idx: int, anchor: Any, char_name: str) -> str:
    return f"""
  {idx}. [TOPIC LEAD-IN]
     The interlocutor should naturally steer the conversation toward this area:
     "{anchor.content}"
     Lead into it organically — through a related observation, a personal experience, or a question about {char_name}'s life.
     {char_name} may not answer immediately; they can reflect, ask a counter-question, or ease into it.
"""


def _build_fragment_instruction(idx: int, anchor: Any, char_name: str) -> str:
    query = anchor.task_data.get("query", "")

    context_block = ""
    if query:
        context_block = f"""
     QA CONTEXT (for your understanding ONLY — DO NOT answer this question in the dialogue,
     DO NOT reveal other answer details. This only helps you understand WHY this fact matters):
     "{query}"
"""

    return f"""
  {idx}. [FACT TO EMBED — MUST APPEAR IN DIALOGUE]
     {char_name} must mention this information during the conversation:
     "{anchor.content}"
     All dates, names, and key terms must appear exactly as written.
     Wrap it in a specific memory — where were they, what did they feel, what was the scene like.
{context_block}"""


def _build_transition_notes(char_name: str) -> str:
    return f"""[SEGMENT: TRANSITION]
- Smoothly shift the conversation topic.
- Can include a pause, a comment about the surroundings, a drink refill, or a new observation.
- Target: 2-4 turns.
"""
