"""
Feature Interpretation Layer — Layer 2
Converts natural-language product change descriptions into structured signals via LLM.
"""
from __future__ import annotations

import json
import os
import hashlib
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


# ── Pydantic Models ────────────────────────────────────────────────

class ChangeType(str, Enum):
    PRICING = 'pricing'
    UI_CHANGE = 'ui_change'
    NEW_FEATURE = 'new_feature'
    REMOVAL = 'removal'
    PERFORMANCE = 'performance'
    POLICY = 'policy'


class SegmentDirection(str, Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'
    UNKNOWN = 'unknown'


class FeatureSignal(BaseModel):
    """Structured representation of a product change's impact signal."""
    feature_name: str = Field(..., description="Short name for the feature change")
    change_type: ChangeType
    affected_dimensions: list[str] = Field(
        ..., description="Subset of: engagement, churn_risk, satisfaction, spend"
    )
    direction_by_segment: dict[str, SegmentDirection] = Field(
        ..., description="Expected direction per segment"
    )
    magnitude_estimate: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


# ── LLM Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a product analytics AI. Given a product change description, output ONLY valid JSON matching this exact schema:

{
  "feature_name": "string - short descriptive name",
  "change_type": "one of: pricing, ui_change, new_feature, removal, performance, policy",
  "affected_dimensions": ["subset of: engagement, churn_risk, satisfaction, spend"],
  "direction_by_segment": {
    "power_user": "positive|negative|neutral|unknown",
    "casual_browser": "positive|negative|neutral|unknown",
    "price_sensitive": "positive|negative|neutral|unknown",
    "early_adopter": "positive|negative|neutral|unknown",
    "enterprise_user": "positive|negative|neutral|unknown"
  },
  "magnitude_estimate": 0.0 to 1.0,
  "confidence": 0.0 to 1.0
}

Rules:
- Output ONLY the JSON object, no markdown, no explanation
- Be conservative — prefer "unknown" over confident wrong answers
- magnitude_estimate should reflect the scale of the change (0.1 = minor tweak, 0.9 = fundamental shift)
- Consider each segment's unique characteristics when predicting direction"""

RETRY_PROMPT = """Your previous response was not valid JSON. Please output ONLY a valid JSON object matching the schema. No markdown formatting, no code blocks, no extra text. Just the raw JSON object starting with { and ending with }."""


# ── In-Memory Cache ────────────────────────────────────────────────

_cache: dict[str, FeatureSignal] = {}


def _cache_key(description: str) -> str:
    return hashlib.md5(description.strip().lower().encode()).hexdigest()


# ── Interpreter ────────────────────────────────────────────────────

def interpret_feature(description: str, api_key: Optional[str] = None) -> FeatureSignal:
    """
    Convert a natural-language product change description into a structured FeatureSignal.
    Uses Gemini API for LLM reasoning with caching and retry logic.
    """
    # Check cache
    key = _cache_key(description)
    if key in _cache:
        print("  [Cache HIT] Returning cached feature signal")
        return _cache[key]

    if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')

    if not api_key or api_key == 'your_gemini_api_key_here':
        print("  [No API Key] Using mock interpretation")
        return _mock_interpret(description)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        # First attempt
        response = model.generate_content(
            f"{SYSTEM_PROMPT}\n\nProduct change description: {description}"
        )
        signal = _parse_response(response.text)

        if signal is None:
            # Retry with correction prompt
            response = model.generate_content(
                f"{SYSTEM_PROMPT}\n\n{RETRY_PROMPT}\n\nProduct change description: {description}"
            )
            signal = _parse_response(response.text)

        if signal is None:
            print("  [Parse Failed] Falling back to mock interpretation")
            signal = _mock_interpret(description)

    except Exception as e:
        print(f"  [API Error] {e} — using mock interpretation")
        signal = _mock_interpret(description)

    _cache[key] = signal
    return signal


def _parse_response(text: str) -> Optional[FeatureSignal]:
    """Try to parse LLM response text into a FeatureSignal."""
    try:
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            cleaned = '\n'.join(lines[1:-1])
        data = json.loads(cleaned)
        return FeatureSignal(**data)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  [Parse Error] {e}")
        return None


def _mock_interpret(description: str) -> FeatureSignal:
    """Rule-based fallback when LLM is unavailable."""
    desc_lower = description.lower()

    # Detect change type
    if any(w in desc_lower for w in ['price', 'cost', 'fee', 'subscription', 'tier']):
        change_type = ChangeType.PRICING
        directions = {
            'power_user': SegmentDirection.NEUTRAL,
            'casual_browser': SegmentDirection.NEGATIVE,
            'price_sensitive': SegmentDirection.NEGATIVE,
            'early_adopter': SegmentDirection.NEUTRAL,
            'enterprise_user': SegmentDirection.NEUTRAL,
        }
        dimensions = ['churn_risk', 'satisfaction', 'spend']
        magnitude = 0.6
    elif any(w in desc_lower for w in ['remov', 'deprecat', 'sunset', 'eliminat']):
        change_type = ChangeType.REMOVAL
        directions = {
            'power_user': SegmentDirection.NEGATIVE,
            'casual_browser': SegmentDirection.NEUTRAL,
            'price_sensitive': SegmentDirection.NEGATIVE,
            'early_adopter': SegmentDirection.NEGATIVE,
            'enterprise_user': SegmentDirection.NEGATIVE,
        }
        dimensions = ['engagement', 'satisfaction', 'churn_risk']
        magnitude = 0.5
    elif any(w in desc_lower for w in ['new', 'add', 'launch', 'introducing', 'ai', 'search']):
        change_type = ChangeType.NEW_FEATURE
        directions = {
            'power_user': SegmentDirection.POSITIVE,
            'casual_browser': SegmentDirection.NEUTRAL,
            'price_sensitive': SegmentDirection.POSITIVE,
            'early_adopter': SegmentDirection.POSITIVE,
            'enterprise_user': SegmentDirection.POSITIVE,
        }
        dimensions = ['engagement', 'satisfaction']
        magnitude = 0.5
    elif any(w in desc_lower for w in ['ui', 'design', 'layout', 'interface', 'theme']):
        change_type = ChangeType.UI_CHANGE
        directions = {
            'power_user': SegmentDirection.NEUTRAL,
            'casual_browser': SegmentDirection.POSITIVE,
            'price_sensitive': SegmentDirection.NEUTRAL,
            'early_adopter': SegmentDirection.POSITIVE,
            'enterprise_user': SegmentDirection.NEUTRAL,
        }
        dimensions = ['engagement', 'satisfaction']
        magnitude = 0.3
    else:
        change_type = ChangeType.POLICY
        directions = {seg: SegmentDirection.UNKNOWN for seg in
                     ['power_user', 'casual_browser', 'price_sensitive', 'early_adopter', 'enterprise_user']}
        dimensions = ['engagement', 'churn_risk', 'satisfaction']
        magnitude = 0.4

    # Increase magnitude if strong language is used
    if any(w in desc_lower for w in ['significant', 'major', 'drastic', 'complete', 'all']):
        magnitude = min(1.0, magnitude + 0.2)

    return FeatureSignal(
        feature_name=description[:60].strip(),
        change_type=change_type,
        affected_dimensions=dimensions,
        direction_by_segment=directions,
        magnitude_estimate=magnitude,
        confidence=0.65,
    )


# ── CLI Test ───────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  FEATURE INTERPRETATION LAYER — Test")
    print("=" * 60)

    test_descriptions = [
        "Increase premium subscription price by 15% and remove free tier export feature",
        "Adding AI-powered search for all tiers",
        "Redesigning the dashboard UI with a new dark mode theme",
    ]

    for desc in test_descriptions:
        print(f"\n  Input: {desc}")
        signal = interpret_feature(desc)
        print(f"  Output: {signal.model_dump_json(indent=2)}")
