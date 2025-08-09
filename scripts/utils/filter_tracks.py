#!/usr/bin/env python3
"""
CSV Filtering Script

This script filters CSV files by searching for specified keywords in the text content.
It's designed to work with music metadata CSV files but can be used with any CSV
that has a 'text' column.

Usage:
    python filter_csv.py input.csv output.csv --keywords "pop" "calm" "melodic"
    python filter_csv.py input.csv output.csv -k "electronic" "synthesizer" --case-sensitive
"""

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def filter_csv_by_keywords(
    input_file: Path,
    output_file: Path,
    keywords: List[str],
    case_sensitive: bool = False,
    text_column: str = "text",
    any_keyword: bool = False,
    anti_keywords: List[str] = None,
) -> None:
    """
    Filter CSV file to include rows containing keywords and exclude rows with anti-keywords.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        keywords: List of keywords to search for
        case_sensitive: Whether to perform case-sensitive search
        text_column: Name of the column to search in
        any_keyword: If True, match any keyword; if False, require all keywords
        anti_keywords: List of keywords that exclude rows if present
    """
    try:
        # Read the input CSV
        logger.info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")

        original_count = len(df)
        logger.info(f"Loaded {original_count:,} rows from {input_file}")

        # Convert keywords to appropriate case for searching
        if not case_sensitive:
            keywords = [kw.lower() for kw in keywords]
            search_text = df[text_column].str.lower()
            if anti_keywords:
                anti_keywords = [kw.lower() for kw in anti_keywords]
        else:
            search_text = df[text_column]

        # Create filter mask for keywords
        if any_keyword:
            logger.info(f"Filtering for ANY of keywords: {keywords} (case_sensitive={case_sensitive})")
            mask = pd.Series([False] * len(df))
            for keyword in keywords:
                keyword_mask = search_text.str.contains(keyword, case=case_sensitive, na=False)
                mask = mask | keyword_mask
                logger.info(f"  After filtering for '{keyword}': {keyword_mask.sum():,} rows contain this keyword")
        else:
            logger.info(f"Filtering for ALL keywords: {keywords} (case_sensitive={case_sensitive})")
            mask = pd.Series([True] * len(df))
            for keyword in keywords:
                keyword_mask = search_text.str.contains(keyword, case=case_sensitive, na=False)
                mask = mask & keyword_mask
                logger.info(f"  After filtering for '{keyword}': {keyword_mask.sum():,} rows contain this keyword")

        # Apply anti-keywords filter (exclude rows containing any anti-keyword)
        if anti_keywords:
            logger.info(f"Excluding rows containing anti-keywords: {anti_keywords}")
            for anti_keyword in anti_keywords:
                anti_mask = search_text.str.contains(anti_keyword, case=case_sensitive, na=False)
                mask = mask & ~anti_mask
                logger.info(f"  Excluded {anti_mask.sum():,} rows containing '{anti_keyword}'")

        # Apply the filter
        filtered_df = df[mask].reset_index(drop=True)
        filtered_count = len(filtered_df)

        # Save the filtered results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(output_file, index=False)

        logger.info(
            f"Filtered results: {filtered_count:,} of {original_count:,} rows kept ({original_count - filtered_count:,} removed)"
        )
        logger.info(f"Saved filtered data to: {output_file}")

        # Show some examples if we have results
        if filtered_count > 0:
            logger.info("First few filtered rows:")
            for idx, row in filtered_df.head(3).iterrows():
                logger.info(f"  [{idx}] {row[text_column][:100]}...")
        else:
            logger.warning("No rows matched all the specified keywords!")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise


def main() -> None:
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Filter CSV files by keywords in text content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/input.csv data/output.csv --keywords "pop" "calm"
  %(prog)s input.csv output.csv -k "electronic" "synthesizer" --case-sensitive
  %(prog)s input.csv output.csv -k "rock" --text-column "description"
  %(prog)s input.csv output.csv -k "pop" "rock" --any-keyword
  %(prog)s input.csv output.csv -k "music" --anti-keywords "sad" "melancholic"
        """,
    )

    parser.add_argument("input_file", type=Path, help="Path to input CSV file")

    parser.add_argument("output_file", type=Path, help="Path to output CSV file")

    parser.add_argument("-k", "--keywords", nargs="+", required=True, help="Keywords to search for")

    parser.add_argument(
        "--anti-keywords", nargs="+", help="Anti-keywords: exclude rows containing any of these keywords"
    )

    parser.add_argument(
        "--any-keyword",
        action="store_true",
        help="Match ANY keyword instead of ALL keywords (default: require all keywords)",
    )

    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Perform case-sensitive keyword matching (default: case-insensitive)",
    )

    parser.add_argument("--text-column", default="text", help="Name of the column to search in (default: 'text')")

    args = parser.parse_args()

    # Validate input file exists
    if not args.input_file.exists():
        parser.error(f"Input file does not exist: {args.input_file}")

    # Run the filtering
    filter_csv_by_keywords(
        input_file=args.input_file,
        output_file=args.output_file,
        keywords=args.keywords,
        case_sensitive=args.case_sensitive,
        text_column=args.text_column,
        any_keyword=args.any_keyword,
        anti_keywords=args.anti_keywords,
    )


if __name__ == "__main__":
    main()
