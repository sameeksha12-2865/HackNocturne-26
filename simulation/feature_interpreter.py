import os
import json
import anthropic
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Optional, Literal

# Attempt to load .env if dotenv is installed, ignore otherwise
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize client. Just relying on ANTHROPIC_API_KEY being in the environment
# The user mentioned pretending there is a valid key.
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "dummy_key"))

class FeatureSignal(BaseModel):
    feature_name: str = Field(description="The name of the feature being proposed short descriptive name.")
    direction_by_segment: Dict[
        Literal['power_user', 'casual_browser', 'price_sensitive', 'early_adopter', 'senior_user'], 
        Literal['positive', 'negative', 'neutral', 'unknown']
    ] = Field(
        description="Predicted impact direction for each segment"
    )
    reasoning: str = Field(description="Brief reasoning for the overall prediction")

SYSTEM_PROMPT = """You are an expert product manager analyzing feature impact.
Analyze the user's proposed feature and output a JSON object adhering to the specified schema.
Segments are: power_user, casual_browser, price_sensitive, early_adopter, senior_user. Be conservative — prefer "unknown" over confident wrong answers.

Return ONLY raw JSON, with no markdown formatting or code blocks.
"""

# Simple in-memory cache to prevent redundant API calls
_cache = {}

def get_feature_signal(feature_description: str) -> Optional[FeatureSignal]:
    """
    Calls Anthropic API to interpret the feature description into a structured FeatureSignal.
    Implements caching and a single retry mechanism on parse failure.
    """
    if feature_description in _cache:
        print("Using cached result for feature signal.")
        return _cache[feature_description]
    
    try:
        schema_str = json.dumps(FeatureSignal.model_json_schema(), indent=2)
    except AttributeError:
        # Fallback for older pydantic versions
        schema_str = FeatureSignal.schema_json(indent=2)
        
    prompt = f"Feature proposal: {feature_description}\n\nPlease output valid JSON adhering exactly to this JSON schema:\n{schema_str}"
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text.strip()
        
        # In case the model responds with Markdown code blocks despite instructions
        if content.startswith("```json"):
            content = content[len("```json"):].strip()
        if content.startswith("```"):
            content = content[len("```"):].strip()
        if content.endswith("```"):
            content = content[:-len("```")].strip()
            
        try:
            parsed = json.loads(content)
            result = FeatureSignal(**parsed)
            _cache[feature_description] = result
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Initial parse or validation failure. Retrying once... Error: {str(e)}")
            correction_prompt = (
                f"Your previous response failed validation with the following error:\n{str(e)}\n\n"
                f"Please correct the JSON so it strictly follows the schema. Return ONLY valid JSON."
            )
            
            retry_response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": correction_prompt}
                ]
            )
            retry_content = retry_response.content[0].text.strip()
            
            # Clean possible markdown block again
            if retry_content.startswith("```json"):
                retry_content = retry_content[len("```json"):].strip()
            if retry_content.startswith("```"):
                retry_content = retry_content[len("```"):].strip()
            if retry_content.endswith("```"):
                retry_content = retry_content[:-len("```")].strip()
                
            retry_parsed = json.loads(retry_content)
            result = FeatureSignal(**retry_parsed)
            _cache[feature_description] = result
            return result
            
    except Exception as e:
        print(f"Error calling LLM or parsing response: {e}")
        return None

if __name__ == '__main__':
    # Simple syntax check when run directly
    print("Feature Interpreter script loaded. Example structure:")
    try:
        print(json.dumps(FeatureSignal.model_json_schema(), indent=2))
    except AttributeError:
        print(FeatureSignal.schema_json(indent=2))
