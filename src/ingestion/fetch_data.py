from pathlib import Path
import nflreadpy as nfl

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = list(range(2008, 2026))  # 2008 through 2025


def fetch_and_save_data():
    print(f"Fetching seasonal stats for {SEASONS[0]}-{SEASONS[-1]}...")
    # summary_level="reg" gives regular-season-aggregated stats,
    # equivalent to what import_seasonal_data() used to return.
    seasonal_df = nfl.load_player_stats(SEASONS, summary_level="reg").to_pandas()
    seasonal_path = RAW_DATA_DIR / "raw_seasonal_stats.csv"
    seasonal_df.to_csv(seasonal_path, index=False)
    print(f"Saved seasonal stats to {seasonal_path}")

    print("Fetching player rosters & metadata...")
    rosters_df = nfl.load_rosters(SEASONS).to_pandas()
    rosters_path = RAW_DATA_DIR / "raw_rosters.csv"
    rosters_df.to_csv(rosters_path, index=False)
    print(f"Saved roster data to {rosters_path}")

    print("Fetching draft pick data...")
    draft_df = nfl.load_draft_picks(SEASONS).to_pandas()
    draft_path = RAW_DATA_DIR / "raw_draft_picks.csv"
    draft_df.to_csv(draft_path, index=False)
    print(f"Saved draft pick data to {draft_path}")


if __name__ == "__main__":
    fetch_and_save_data()