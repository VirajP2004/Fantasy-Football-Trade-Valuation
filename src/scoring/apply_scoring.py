import yaml
import pandas as pd
from pathlib import Path

def load_scoring_rules(config_path="config/league_scoring_rules.yaml"):
    """Loads league scoring rules from the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_rule(rules, category, key):
    """Safely retrieves a rule from YAML or raises an error if missing."""
    try:
        return rules[category][key]
    except KeyError:
        raise KeyError(f"Missing scoring rule for [{category}][{key}] in league_scoring_rules.yaml!")

def compute_fantasy_points(df, rules):
    """
    Applies league scoring rules dynamically, mapping dataset columns to YAML configuration keys.
    """
    fp = 0.0

    # Passing (Dataset col -> YAML key mapping)
    if "passing_yards" in df.columns:
        fp += df["passing_yards"].fillna(0) * get_rule(rules, "passing", "passing_yds")
    if "passing_tds" in df.columns:
        fp += df["passing_tds"].fillna(0) * get_rule(rules, "passing", "passing_td")
    if "passing_interceptions" in df.columns:
        fp += df["passing_interceptions"].fillna(0) * get_rule(rules, "passing", "interceptions")
    if "passing_2pt_conversions" in df.columns:
        fp += df["passing_2pt_conversions"].fillna(0) * get_rule(rules, "passing", "passing_two_pt")

    # Rushing
    if "rushing_yards" in df.columns:
        fp += df["rushing_yards"].fillna(0) * get_rule(rules, "rushing", "rushing_yds")
    if "rushing_tds" in df.columns:
        fp += df["rushing_tds"].fillna(0) * get_rule(rules, "rushing", "rushing_td")
    if "rushing_2pt_conversions" in df.columns:
        fp += df["rushing_2pt_conversions"].fillna(0) * get_rule(rules, "rushing", "rushing_two_pt")

    # Receiving
    if "receptions" in df.columns:
        fp += df["receptions"].fillna(0) * get_rule(rules, "receiving", "receptions")
    if "receiving_yards" in df.columns:
        fp += df["receiving_yards"].fillna(0) * get_rule(rules, "receiving", "receiving_yds")
    if "receiving_tds" in df.columns:
        fp += df["receiving_tds"].fillna(0) * get_rule(rules, "receiving", "receiving_td")
    if "receiving_2pt_conversions" in df.columns:
        fp += df["receiving_2pt_conversions"].fillna(0) * get_rule(rules, "receiving", "receiving_two_pt")

    # Fumbles — flat penalty regardless of position (QB/RB/WR/TE all lose
    # the same points per fumble lost). The raw dataset has no column
    # literally named "fumbles_lost"; the pre-aggregated sum across
    # sack/rushing/receiving fumbles lost is "fumbles_lost_total".
    if "fumbles_lost_total" in df.columns:
        fp += df["fumbles_lost_total"].fillna(0) * get_rule(rules, "fumbles", "fumbles_lost")

    return fp

def main():
    print("Loading raw seasonal stats...")
    raw_path = Path("data/raw/raw_seasonal_stats.csv")
    if not raw_path.exists():
        print("Error: Raw seasonal stats not found. Please run fetch_data.py first.")
        return

    df = pd.read_csv(raw_path)
    rules = load_scoring_rules()

    print("Applying custom league scoring rules...")
    df["custom_fantasy_points"] = compute_fantasy_points(df, rules)

    # Save intermediate scored dataset
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scored_seasonal_stats.csv"
    
    df.to_csv(output_path, index=False)
    print(f"Scored stats successfully saved to {output_path}")
    
    # Print sample top players by custom points
    sample_view = df[["player_display_name", "season", "position", "custom_fantasy_points"]].sort_values(by="custom_fantasy_points", ascending=False)
    print(sample_view.head(10))

if __name__ == "__main__":
    main()