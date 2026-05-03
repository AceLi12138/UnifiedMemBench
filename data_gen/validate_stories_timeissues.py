"""
Stories V3 Data Validator

Validate data-quality issues in stories_v3.json.

Checks:
1. Year format issues: non-integer values, out-of-range values, or stray text.
2. Causality issues: missing references or self-references.
3. Event ID issues: duplicates or malformed IDs.
4. Time description issues: missing or inconsistent values.
5. Empty values or missing fields.
6. Character-encoding issues.
"""

import json
import re
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


def check_year_format(year_value) -> List[str]:
    """Check year format."""
    issues = []
    
    if year_value is None:
        issues.append("Year is empty")
        return issues
    
    if isinstance(year_value, str):
        # Detect stray non-numeric text.
        if not year_value.strip().isdigit():
            issues.append(f"Year contains non-numeric characters: '{year_value}'")
        else:
            year_int = int(year_value)
            if year_int < 1900 or year_int > 2100:
                issues.append(f"Year is outside the expected range: {year_int}")
    elif isinstance(year_value, int):
        if year_value < 1900 or year_value > 2100:
            issues.append(f"Year is outside the expected range: {year_value}")
        if year_value > 10000:
            issues.append(f"Year may contain an input error with extra digits: {year_value}")
    else:
        issues.append(f"Unexpected year type: {type(year_value).__name__} = {year_value}")
    
    return issues


def check_event_id(event_id: str, year_value, all_ids: Set[str]) -> List[str]:
    """Check event ID format."""
    issues = []
    
    if not event_id:
        issues.append("Event ID is empty")
        return issues
    
    # Check duplicates.
    if event_id in all_ids:
        issues.append(f"Duplicate event ID: {event_id}")
    
    # Check format (expected: YYYY_EX).
    pattern = r'^\d{4}_E\d+$'
    if not re.match(pattern, event_id):
        issues.append(f"Malformed event ID: {event_id} (expected format: YYYY_EX)")
    
    # Check whether the ID year matches the year entry.
    if event_id and "_" in event_id:
        id_year = event_id.split("_")[0]
        try:
            id_year_int = int(id_year)
            year_int = int(year_value) if year_value else 0
            if id_year_int != year_int:
                issues.append(f"Event ID year ({id_year_int}) does not match parent year ({year_int})")
        except:
            pass
    
    return issues


def parse_time_of_year(time_of_year: str) -> int:
    """
    Parse time_of_year into a comparable month value from 1 to 12.

    Return 0 when the value cannot be parsed.
    """
    if not time_of_year or not isinstance(time_of_year, str):
        return 0
    
    lower_text = time_of_year.lower()
    
    # Month mapping.
    month_map = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }
    
    # Season mapping using the middle month of each season.
    season_map = {
        "winter": 1,   # Winter (Dec-Feb), represented by January.
        "spring": 4,   # Spring (Mar-May), represented by April.
        "summer": 7,   # Summer (Jun-Aug), represented by July.
        "fall": 10,    # Fall (Sep-Nov), represented by October.
        "autumn": 10,
    }
    
    # Early/mid/late adjustment within the same season.
    time_modifier = 0
    if "early" in lower_text:
        time_modifier = -0.3
    elif "late" in lower_text:
        time_modifier = 0.3
    elif "mid" in lower_text:
        time_modifier = 0
    
    # First try explicit month names.
    for month_name, month_num in month_map.items():
        if month_name in lower_text:
            return month_num
    
    # Then try season names.
    for season_name, month_num in season_map.items():
        if season_name in lower_text:
            return month_num
    
    return 0


def check_causality(event: Dict, all_event_ids: Set[str], event_time_map: Dict[str, Tuple[int, int]] = None) -> List[str]:
    """
    Check causality references.
    
    Args:
        event: Current event.
        all_event_ids: Set of all event IDs.
        event_time_map: Mapping from event ID to (year, month), used for
            finer-grained temporal ordering checks.
    """
    issues = []
    
    caused_by = event.get("caused_by_event_ids", [])
    event_id = event.get("event_id", "Unknown")
    
    if caused_by is None:
        return issues
    
    if not isinstance(caused_by, list):
        issues.append(f"caused_by_event_ids is not a list: {type(caused_by).__name__}")
        return issues
    
    for cause_id in caused_by:
        # Self-reference.
        if cause_id == event_id:
            issues.append(f"Self-reference: {event_id} -> itself")
            continue
        
        # Reference to a missing event.
        if cause_id not in all_event_ids:
            issues.append(f"Reference to missing event: {event_id} -> {cause_id}")
            continue
        
        # Temporal ordering check using the precise time map.
        if event_time_map and cause_id in event_time_map and event_id in event_time_map:
            cause_year, cause_month = event_time_map[cause_id]
            effect_year, effect_month = event_time_map[event_id]
            
            # The cause happens after the effect.
            if cause_year > effect_year:
                issues.append(f"Causality inversion across years: {event_id} ({effect_year}) is caused by {cause_id} ({cause_year})")
            elif cause_year == effect_year and cause_month > effect_month:
                issues.append(f"Causality inversion within a year: {event_id} ({effect_year}-{effect_month:02d}) is caused by {cause_id} ({cause_year}-{cause_month:02d})")
        else:
            # Fall back to a year-only check.
            if "_" in cause_id and "_" in event_id:
                try:
                    cause_year = int(cause_id.split("_")[0])
                    effect_year = int(event_id.split("_")[0])
                    if cause_year > effect_year:
                        issues.append(f"Causality inversion across years: {event_id} ({effect_year}) is caused by {cause_id} ({cause_year})")
                except:
                    pass
    
    return issues


def check_time_of_year(time_of_year) -> List[str]:
    """Check time description."""
    issues = []
    
    if not time_of_year:
        issues.append("time_of_year is empty")
        return issues
    
    if not isinstance(time_of_year, str):
        issues.append(f"Unexpected time_of_year type: {type(time_of_year).__name__}")
        return issues
    
    # Check whether the value contains a valid month or season keyword.
    valid_keywords = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "winter", "spring", "summer", "fall", "autumn",
        "early", "mid", "late"
    ]
    
    lower_text = time_of_year.lower()
    has_valid = any(kw in lower_text for kw in valid_keywords)
    
    if not has_valid:
        issues.append(f"time_of_year is not recognized: '{time_of_year}'")
    
    return issues


def check_required_fields(event: Dict) -> List[str]:
    """Check required fields."""
    issues = []
    
    required = ["event_id", "category", "description"]
    for field in required:
        value = event.get(field)
        if value is None:
            issues.append(f"Missing required field: {field}")
        elif isinstance(value, str) and not value.strip():
            issues.append(f"Required field is an empty string: {field}")
    
    return issues


def check_character(character: Dict, char_index: int) -> Dict:
    """Check all validation issues for one character."""
    result = {
        "character_name": character.get("character_name", f"Unknown_{char_index}"),
        "issues": [],
        "warnings": [],
        "stats": {}
    }
    
    char_name = result["character_name"]
    
    # Check character name.
    if not char_name or char_name.startswith("Unknown"):
        result["issues"].append("Missing or invalid character_name")
    
    chronology = character.get("chronology", [])
    if not chronology:
        result["issues"].append("chronology is empty")
        return result
    
    # Collect all event IDs and time information in the first pass.
    all_event_ids = set()
    event_time_map = {}  # event_id -> (year, month)
    
    for year_entry in chronology:
        year_value = year_entry.get("year")
        try:
            year_int = int(year_value) if year_value else 0
        except:
            year_int = 0
            
        for event in year_entry.get("events", []):
            eid = event.get("event_id")
            if eid:
                all_event_ids.add(eid)
                # Parse time.
                time_of_year = event.get("time_of_year", "")
                month = parse_time_of_year(time_of_year)
                event_time_map[eid] = (year_int, month)
    
    # Statistics.
    total_events = 0
    seen_ids = set()
    year_values = []
    
    # Detailed checks.
    for year_idx, year_entry in enumerate(chronology):
        year_value = year_entry.get("year")
        year_values.append(year_value)
        
        # Year issues.
        year_issues = check_year_format(year_value)
        for issue in year_issues:
            result["issues"].append(f"[Year {year_value}] {issue}")
        
        events = year_entry.get("events", [])
        if not events:
            result["warnings"].append(f"[Year {year_value}] has no events")
            continue
        
        for event_idx, event in enumerate(events):
            total_events += 1
            event_id = event.get("event_id", f"Unknown_{year_idx}_{event_idx}")
            
            # Event ID issues.
            id_issues = check_event_id(event_id, year_value, seen_ids)
            for issue in id_issues:
                result["issues"].append(f"[{event_id}] {issue}")
            seen_ids.add(event_id)
            
            # Causality issues using the time map.
            causality_issues = check_causality(event, all_event_ids, event_time_map)
            for issue in causality_issues:
                result["issues"].append(f"[{event_id}] {issue}")
            
            # Time description issues.
            time_issues = check_time_of_year(event.get("time_of_year"))
            for issue in time_issues:
                result["warnings"].append(f"[{event_id}] {issue}")
            
            # Required field issues.
            field_issues = check_required_fields(event)
            for issue in field_issues:
                result["issues"].append(f"[{event_id}] {issue}")
    
    # Convert year values to strings for safe display
    year_strs = [str(y) for y in year_values if y is not None]
    result["stats"] = {
        "total_years": len(chronology),
        "total_events": total_events,
        "year_range": f"{year_strs[0] if year_strs else 'N/A'} - {year_strs[-1] if year_strs else 'N/A'}",
        "issue_count": len(result["issues"]),
        "warning_count": len(result["warnings"])
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate stories_v3.json data quality")
    parser.add_argument("--input_file", type=str, 
                        default="./output/stories_v3.json",
                        help="Input file to validate")
    parser.add_argument("--output_report", type=str, 
                        default=None,
                        help="Output validation report")
    parser.add_argument("--verbose", action="store_true",
                        help="Print all issues for each character")
    
    args = parser.parse_args()
    
    if not args.output_report:
        args.output_report = args.input_file.replace(".json", "_validation_report.json")
    # Load data (support both JSON array and JSONL)
    print(f"Loading: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            characters = json.load(f)
        else:
            # JSONL format
            characters = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(characters)} characters\n")
    
    # Validate all characters
    results = []
    summary = {
        "total_characters": len(characters),
        "characters_with_issues": 0,
        "characters_with_warnings": 0,
        "total_issues": 0,
        "total_warnings": 0,
        "issue_types": Counter(),
        "problematic_characters": []
    }
    
    for i, char in enumerate(characters):
        result = check_character(char, i)
        results.append(result)
        
        if result["issues"]:
            summary["characters_with_issues"] += 1
            summary["total_issues"] += len(result["issues"])
            summary["problematic_characters"].append(result["character_name"])
            
            # Categorize issues
            for issue in result["issues"]:
                if "Causality" in issue or "Reference" in issue or "caused_by_event_ids" in issue:
                    summary["issue_types"]["Causality issues"] += 1
                elif "ID" in issue:
                    summary["issue_types"]["Event ID issues"] += 1
                elif "Year" in issue or "year" in issue:
                    summary["issue_types"]["Year format issues"] += 1
                elif "Missing" in issue or "empty" in issue:
                    summary["issue_types"]["Missing field issues"] += 1
                else:
                    summary["issue_types"]["Other issues"] += 1
        
        if result["warnings"]:
            summary["characters_with_warnings"] += 1
            summary["total_warnings"] += len(result["warnings"])
    
    # Print summary
    print("="*70)
    print("📊 STORIES V3 VALIDATION SUMMARY")
    print("="*70)
    print(f"\nTotal characters: {summary['total_characters']}")
    print(f"Characters with issues: {summary['characters_with_issues']} ({100*summary['characters_with_issues']/max(1,summary['total_characters']):.1f}%)")
    print(f"Total issues: {summary['total_issues']}")
    print(f"Total warnings: {summary['total_warnings']}")
    
    print("\nIssue type distribution:")
    for issue_type, count in summary["issue_types"].most_common():
        print(f"  - {issue_type}: {count}")
    
    # Print problematic characters
    if summary["problematic_characters"]:
        print("\nProblematic characters (first 20):")
        for name in summary["problematic_characters"][:20]:
            # Find and print their issues
            for r in results:
                if r["character_name"] == name:
                    print(f"\n  [{name}] ({r['stats']['issue_count']} issues)")
                    for issue in r["issues"][:5]:
                        print(f"    - {issue}")
                    if len(r["issues"]) > 5:
                        print(f"    ... {len(r['issues'])-5} more issues")
                    break
    
    # Save full report
    report = {
        "summary": summary,
        "details": results
    }
    
    with open(args.output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nFull report saved to: {args.output_report}")
    
    # Return exit code based on issues
    if summary["total_issues"] > 0:
        print(f"\nFound {summary['total_issues']} issues that need fixing")
        return 1
    else:
        print("\nValidation passed with no issues")
        return 0


if __name__ == "__main__":
    exit(main())
