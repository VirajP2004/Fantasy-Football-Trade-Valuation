import pandas as pd

# Load seasonal stats
df = pd.read_csv("data/raw/raw_seasonal_stats.csv")

print(f"Total rows in seasonal stats: {len(df)}")
print("\nSample columns available:")
print(df.columns[:15].tolist())

# Spot-check 2023 season for a player using the correct 'passing_yards' column
if "season" in df.columns and "player_display_name" in df.columns:
    subset = df[(df["season"] == 2023) & (df["passing_yards"] > 4000)]
    print("\nSample 2023 4000+ yard passers:")
    print(subset[["player_display_name", "season", "passing_yards"]].head())
else:
    print("\nColumns check: season or player name columns missing.")