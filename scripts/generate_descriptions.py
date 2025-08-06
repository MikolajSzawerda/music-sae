import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ExpandedDescriptions(BaseModel):
    """Represents expanded descriptions for a musical feature."""

    variations: List[str] = Field(
        ...,
        description="List of 10 diverse but similar descriptions for the musical feature",
        min_items=10,
        max_items=10,
    )


def create_expansion_prompt(original_description: str) -> str:
    """Create a prompt for expanding a feature description."""
    return f"""
You are given this description of a musical feature:
"{original_description}"

Generate exactly 10 alternative descriptions that:
1. Describe the SAME musical concept/feature
2. Use DIFFERENT wording and vocabulary
3. Vary in length and style (some shorter, some longer)
4. Maintain the core musical meaning
5. Are diverse but semantically equivalent

Focus on musical characteristics like:
- Instruments mentioned
- Tempo/rhythm descriptions  
- Mood/atmosphere
- Musical style/genre
- Production techniques
- Melodic/harmonic qualities

Each description should be a complete, standalone sentence that could replace the original.
Avoid simply rearranging the same words - use synonyms, alternative phrasings, and different perspectives.
""".strip()


def expand_feature_description(
    feature_name: str, original_description: str, model_name: str = "gemini-1.5-flash"
) -> List[str]:
    """Use Gemini to expand a single feature description."""
    api_key = os.getenv("GENMINI_API")
    if not api_key:
        raise ValueError("GENMINI_API environment variable not set")

    client = genai.Client(api_key=api_key)

    prompt = create_expansion_prompt(original_description)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": ExpandedDescriptions,
            },
        )

        result = ExpandedDescriptions.model_validate(json.loads(response.text))
        return result.variations

    except Exception as e:
        logger.error(f"Error expanding description for {feature_name}: {e}")
        # Return the original description repeated if there's an error
        return [original_description] * 10


def load_feature_descriptions(file_path: Path) -> Dict[str, str]:
    """Load feature descriptions from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_expanded_descriptions(expanded: Dict[str, List[str]], output_path: Path):
    """Save expanded descriptions to JSON file."""
    with open(output_path, "w") as f:
        json.dump(expanded, f, indent=2)
    logger.info(f"Expanded descriptions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Expand feature descriptions using Gemini")
    parser.add_argument("input_file", type=Path, help="Input JSON file with feature descriptions")
    parser.add_argument(
        "--output",
        type=Path,
        default="expanded_descriptions.json",
        help="Output JSON file path (default: expanded_descriptions.json)",
    )
    parser.add_argument(
        "--gemini-model", default="gemini-1.5-flash", help="Gemini model to use (default: gemini-1.5-flash)"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Load original descriptions
    logger.info(f"Loading feature descriptions from: {args.input_file}")
    feature_descriptions = load_feature_descriptions(args.input_file)
    logger.info(f"Found {len(feature_descriptions)} features to expand")

    # Expand each feature description
    expanded_descriptions = {}

    for feature_name, original_desc in tqdm(feature_descriptions.items(), desc="Expanding descriptions"):
        logger.info(f"Expanding {feature_name}")

        expanded_variations = expand_feature_description(feature_name, original_desc, args.gemini_model)

        expanded_descriptions[feature_name] = expanded_variations

        # Log first few variations for verification
        logger.debug(f"{feature_name} - Original: {original_desc}")
        for i, variation in enumerate(expanded_variations[:3]):
            logger.debug(f"{feature_name} - Variation {i + 1}: {variation}")

    # Save results
    save_expanded_descriptions(expanded_descriptions, args.output)

    # Print summary
    total_descriptions = len(feature_descriptions) * 10
    print(f"\n{'=' * 60}")
    print("EXPANSION SUMMARY")
    print("=" * 60)
    print(f"Features processed: {len(feature_descriptions)}")
    print(f"Total descriptions generated: {total_descriptions}")
    print("Descriptions per feature: 10")
    print(f"Output saved to: {args.output}")

    # Show example
    if expanded_descriptions:
        example_feature = next(iter(expanded_descriptions.keys()))
        print(f"\nExample for {example_feature}:")
        print(f"Original: {feature_descriptions[example_feature]}")
        print("Variations:")
        for i, variation in enumerate(expanded_descriptions[example_feature][:3]):
            print(f"  {i + 1}. {variation}")
        print("  ... (7 more variations)")

    logger.info("Feature description expansion complete!")


if __name__ == "__main__":
    main()
