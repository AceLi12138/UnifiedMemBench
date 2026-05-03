"""
Strategy A: character voice system (Speech Profile Generator).

Generate a speaking-style description from MBTI, demographics, tone_of_voice,
and hobbies, then inject it into the system prompt instead of the generic V7
"Be verbose. Each turn should be 50-150 words." instruction.

Allowed safe fields:
  - original_persona.current_mbti
  - original_persona.demographics (age, gender, location, occupation)
  - original_persona.current_profile.tone_of_voice
  - original_persona.current_profile.hobbies

Do not use fields that may contain event information and leak answers:
  - personality_summary
  - recent_anxiety_or_goal
  - evolution_timeline
"""

from typing import Dict, Any, List


# Baseline speaking-style map for the 16 MBTI types.
MBTI_SPEECH_STYLES: Dict[str, Dict[str, str]] = {
    # Analysts
    "INTJ": {
        "verbosity": "concise",
        "formality": "formal",
        "metaphor_tendency": "occasional",
        "emotional_expression": "restrained",
        "humor_frequency": "rare",
    },
    "INTP": {
        "verbosity": "moderate",
        "formality": "neutral",
        "metaphor_tendency": "occasional",
        "emotional_expression": "restrained",
        "humor_frequency": "occasional",
    },
    "ENTJ": {
        "verbosity": "moderate",
        "formality": "formal",
        "metaphor_tendency": "rare",
        "emotional_expression": "moderate",
        "humor_frequency": "rare",
    },
    "ENTP": {
        "verbosity": "verbose",
        "formality": "casual",
        "metaphor_tendency": "frequent",
        "emotional_expression": "expressive",
        "humor_frequency": "frequent",
    },
    # Diplomats
    "INFJ": {
        "verbosity": "moderate",
        "formality": "neutral",
        "metaphor_tendency": "frequent",
        "emotional_expression": "moderate",
        "humor_frequency": "occasional",
    },
    "INFP": {
        "verbosity": "moderate",
        "formality": "casual",
        "metaphor_tendency": "frequent",
        "emotional_expression": "expressive",
        "humor_frequency": "occasional",
    },
    "ENFJ": {
        "verbosity": "verbose",
        "formality": "neutral",
        "metaphor_tendency": "occasional",
        "emotional_expression": "expressive",
        "humor_frequency": "occasional",
    },
    "ENFP": {
        "verbosity": "verbose",
        "formality": "casual",
        "metaphor_tendency": "frequent",
        "emotional_expression": "expressive",
        "humor_frequency": "frequent",
    },
    # Sentinels
    "ISTJ": {
        "verbosity": "concise",
        "formality": "formal",
        "metaphor_tendency": "rare",
        "emotional_expression": "restrained",
        "humor_frequency": "rare",
    },
    "ISFJ": {
        "verbosity": "moderate",
        "formality": "neutral",
        "metaphor_tendency": "rare",
        "emotional_expression": "moderate",
        "humor_frequency": "rare",
    },
    "ESTJ": {
        "verbosity": "concise",
        "formality": "formal",
        "metaphor_tendency": "rare",
        "emotional_expression": "restrained",
        "humor_frequency": "occasional",
    },
    "ESFJ": {
        "verbosity": "verbose",
        "formality": "neutral",
        "metaphor_tendency": "occasional",
        "emotional_expression": "expressive",
        "humor_frequency": "occasional",
    },
    # Explorers
    "ISTP": {
        "verbosity": "concise",
        "formality": "casual",
        "metaphor_tendency": "rare",
        "emotional_expression": "restrained",
        "humor_frequency": "occasional",
    },
    "ISFP": {
        "verbosity": "moderate",
        "formality": "casual",
        "metaphor_tendency": "occasional",
        "emotional_expression": "moderate",
        "humor_frequency": "rare",
    },
    "ESTP": {
        "verbosity": "concise",
        "formality": "casual",
        "metaphor_tendency": "rare",
        "emotional_expression": "moderate",
        "humor_frequency": "frequent",
    },
    "ESFP": {
        "verbosity": "verbose",
        "formality": "casual",
        "metaphor_tendency": "occasional",
        "emotional_expression": "expressive",
        "humor_frequency": "frequent",
    },
}

VERBOSITY_TURN_GUIDE = {
    "concise": "Keep responses short and direct (30-80 words per turn). Avoid over-explaining.",
    "moderate": "Respond with moderate detail (60-120 words per turn). Balance depth with brevity.",
    "verbose": "Give rich, detailed responses (80-150 words per turn). Elaborate with stories and examples.",
}

FORMALITY_GUIDE = {
    "casual": "Use everyday, conversational language. Contractions, slang, and sentence fragments are fine.",
    "neutral": "Use clear, approachable language. Neither stiff nor overly casual.",
    "formal": "Use precise, well-structured language. Complete sentences, measured word choice.",
}

METAPHOR_GUIDE = {
    "rare": "Speak plainly and literally. Avoid flowery language or extended metaphors.",
    "occasional": "Use the occasional comparison or image when it helps, but don't overdo it.",
    "frequent": "Naturally weave in imagery, analogies, and sensory descriptions when expressing ideas.",
}

EMOTION_GUIDE = {
    "restrained": "Keep emotions understated. Show feeling through actions and facts rather than direct declarations.",
    "moderate": "Express emotions naturally but without excess. A balanced emotional register.",
    "expressive": "Be open and animated. Show enthusiasm, concern, or nostalgia freely when it fits.",
}

HUMOR_GUIDE = {
    "rare": "Maintain a serious, thoughtful tone. Humor is not a natural mode.",
    "occasional": "Drop in dry wit or gentle humor when the moment calls for it.",
    "frequent": "Use humor freely — self-deprecating jokes, playful teasing, funny asides.",
}

FORMAL_OCCUPATIONS = {
    "professor", "lawyer", "attorney", "judge", "diplomat", "surgeon",
    "physician", "doctor", "executive", "director", "principal",
    "therapist", "counselor", "accountant", "architect", "scientist",
}

CASUAL_OCCUPATIONS = {
    "bartender", "barista", "hairstylist", "tattoo artist", "musician",
    "dj", "skateboarder", "comedian", "street artist", "gig worker",
    "driver", "mechanic", "cook", "chef", "trainer", "coach",
}


def _adjust_for_demographics(style: Dict[str, str], demographics: Dict[str, Any]) -> Dict[str, str]:
    """Adjust speaking style using demographic attributes."""
    style = dict(style)
    age = demographics.get("age", 30)
    occupation = (demographics.get("occupation") or "").lower()

    if age < 25:
        if style["formality"] == "formal":
            style["formality"] = "neutral"
        elif style["formality"] == "neutral":
            style["formality"] = "casual"
        if style["humor_frequency"] == "rare":
            style["humor_frequency"] = "occasional"

    if age > 55:
        if style["formality"] == "casual":
            style["formality"] = "neutral"

    occ_tokens = set(occupation.replace("-", " ").replace("/", " ").split())
    if occ_tokens & FORMAL_OCCUPATIONS:
        if style["formality"] == "casual":
            style["formality"] = "neutral"
    if occ_tokens & CASUAL_OCCUPATIONS:
        if style["formality"] == "formal":
            style["formality"] = "neutral"

    return style


def _collect_text_parts(value: Any) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            parts.extend(_collect_text_parts(item))
        return parts
    if isinstance(value, dict):
        parts: List[str] = []
        for key, item in value.items():
            if isinstance(item, bool):
                if item and isinstance(key, str):
                    text = key.replace("_", " ").strip()
                    if text:
                        parts.append(text)
                continue
            parts.extend(_collect_text_parts(item))
        return parts
    return []


def _hobbies_to_text(hobbies: Any) -> str:
    if isinstance(hobbies, str):
        return hobbies.strip()

    parts = _collect_text_parts(hobbies)
    deduped: List[str] = []
    seen = set()
    for part in parts:
        if part not in seen:
            deduped.append(part)
            seen.add(part)
    return "; ".join(deduped)


def generate_speech_profile(char_profile: Dict[str, Any]) -> str:
    """
    Generate a speaking-style block for the system prompt.

    Args:
        char_profile: One character object from char_map[name] in stories_v4.json.

    Returns:
        Formatted text that can be injected into the system prompt.
    """
    persona = char_profile.get("original_persona", {})
    mbti = persona.get("current_mbti", "INFP")
    demographics = persona.get("demographics", {})
    profile = persona.get("current_profile", {})
    tone_of_voice = profile.get("tone_of_voice", "")
    hobbies = profile.get("hobbies", "")

    base_style = MBTI_SPEECH_STYLES.get(mbti, MBTI_SPEECH_STYLES["INFP"])
    style = _adjust_for_demographics(base_style, demographics)

    lines = [
        "[CHARACTER SPEECH STYLE]",
        VERBOSITY_TURN_GUIDE[style["verbosity"]],
        FORMALITY_GUIDE[style["formality"]],
        METAPHOR_GUIDE[style["metaphor_tendency"]],
        EMOTION_GUIDE[style["emotional_expression"]],
        HUMOR_GUIDE[style["humor_frequency"]],
    ]

    if tone_of_voice:
        lines.append(f"Authentic voice reference: {tone_of_voice}")

    hobbies_text = _hobbies_to_text(hobbies)
    if hobbies_text:
        lines.append(
            f"The character's interests include: {hobbies_text[:200]}. "
            "These may naturally color their vocabulary and references."
        )

    return "\n".join(lines)
