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

    # Kicking — field goals bucketed by distance, PATs, and misses.
    if "fg_made_0_19" in df.columns:
        fp += df["fg_made_0_19"].fillna(0) * get_rule(rules, "kicking", "fg_0_19")
    if "fg_made_20_29" in df.columns:
        fp += df["fg_made_20_29"].fillna(0) * get_rule(rules, "kicking", "fg_20_29")
    if "fg_made_30_39" in df.columns:
        fp += df["fg_made_30_39"].fillna(0) * get_rule(rules, "kicking", "fg_30_39")
    if "fg_made_40_49" in df.columns:
        fp += df["fg_made_40_49"].fillna(0) * get_rule(rules, "kicking", "fg_40_49")

    # league_scoring_rules.yaml has one combined "fg_50_plus" bucket, but
    # the raw dataset splits it into two distance columns — sum both.
    long_fgs_made = pd.Series(0, index=df.index)
    has_long_fg_data = False
    if "fg_made_50_59" in df.columns:
        long_fgs_made = long_fgs_made + df["fg_made_50_59"].fillna(0)
        has_long_fg_data = True
    if "fg_made_60_" in df.columns:
        long_fgs_made = long_fgs_made + df["fg_made_60_"].fillna(0)
        has_long_fg_data = True
    if has_long_fg_data:
        fp += long_fgs_made * get_rule(rules, "kicking", "fg_50_plus")

    if "pat_made" in df.columns:
        fp += df["pat_made"].fillna(0) * get_rule(rules, "kicking", "pat_made")

    # A BLOCKED kick isn't itemized as its own rule in
    # league_scoring_rules.yaml, but it's still a failed attempt from the
    # kicker's perspective — folded into that kick type's "missed" count.
    # Verified against this league's actual live Sleeper scoring: a
    # kicker with 1 blocked FG and full-season roster coverage matched
    # Sleeper's real season total exactly only once fg_blocked was
    # counted alongside fg_missed (same confirmed for pat_blocked /
    # pat_missed via a separate kicker) — leaving blocks unscored would
    # silently undercount the penalty by 1 point per block, the same
    # class of bug as the fumbles_lost column mismatch.
    fg_missed_total = pd.Series(0, index=df.index)
    has_fg_missed_data = False
    if "fg_missed" in df.columns:
        fg_missed_total = fg_missed_total + df["fg_missed"].fillna(0)
        has_fg_missed_data = True
    if "fg_blocked" in df.columns:
        fg_missed_total = fg_missed_total + df["fg_blocked"].fillna(0)
        has_fg_missed_data = True
    if has_fg_missed_data:
        fp += fg_missed_total * get_rule(rules, "kicking", "fg_missed")

    pat_missed_total = pd.Series(0, index=df.index)
    has_pat_missed_data = False
    if "pat_missed" in df.columns:
        pat_missed_total = pat_missed_total + df["pat_missed"].fillna(0)
        has_pat_missed_data = True
    if "pat_blocked" in df.columns:
        pat_missed_total = pat_missed_total + df["pat_blocked"].fillna(0)
        has_pat_missed_data = True
    if has_pat_missed_data:
        fp += pat_missed_total * get_rule(rules, "kicking", "pat_missed")

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