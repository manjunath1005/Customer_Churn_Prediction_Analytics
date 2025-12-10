import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "telco_data"

PROCESSED_DIR = os.path.join("data", "processed")
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

SUMMARY_CSV_PATH = os.path.join(PROCESSED_DIR, "analysis_summary.csv")

# Toggle plotting
GENERATE_PLOTS = True


#  Utilities 
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError("Missing SUPABASE_URL or SUPABASE_KEY in environment (.env)")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_table_from_supabase(supabase, table_name: str) -> pd.DataFrame:
    """
    Fetches entire table from Supabase and returns a DataFrame.
    Handles different client response shapes.
    """
    resp = supabase.table(table_name).select("*").execute()

    # Different versions return objects or dicts
    if hasattr(resp, "data"):
        data = resp.data
    elif isinstance(resp, dict):
        data = resp.get("data", [])
    else:
        # Try to convert to dict-like
        try:
            data = resp[0]
        except Exception:
            data = []

    # If data is empty, return empty DF
    if not data:
        return pd.DataFrame()

    # Convert to DataFrame; ensure columns are normalized (strip)
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    return df


def normalize_churn_values(df: pd.DataFrame, churn_col_candidates=("Churn", "churn")) -> pd.Series:
    """Return a clean churn series with values 'Yes'/'No' (or NaN)"""
    col = None
    for c in churn_col_candidates:
        if c in df.columns:
            col = c
            break
    if col is None:
        # create a column if missing
        return pd.Series([np.nan] * len(df), name="Churn")

    s = df[col].astype(str).str.strip().str.lower()
    # common mappings
    s = s.replace({"yes": "Yes", "y": "Yes", "no": "No", "n": "No", "true": "Yes", "false": "No"})
    s = s.where(~s.isin(["nan", "none", "na", ""]), other=np.nan)
    # Capitalize canonical values
    s = s.map(lambda x: x.capitalize() if pd.notna(x) else x)
    s.name = "Churn"
    return s


#  Analysis functions 
def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute the requested metrics and return a dictionary."""
    metrics = {}
    # churn series normalized
    churn = normalize_churn_values(df)
    total = len(churn.dropna()) if churn.dropna().shape[0] > 0 else len(df)

    # churn percentage (count of 'Yes' / total rows)
    yes_count = (churn == "Yes").sum()
    total_for_pct = len(df) if total == 0 else total
    churn_pct = round(100 * yes_count / total_for_pct, 3) if total_for_pct > 0 else 0.0
    metrics["churn_percentage"] = churn_pct
    metrics["churn_yes_count"] = int(yes_count)
    metrics["total_rows_counted_for_churn"] = int(total_for_pct)

    # Average monthly charges per contract
    if "MonthlyCharges" in df.columns and "Contract" in df.columns:
        # coerce to numeric
        df["MonthlyCharges_num"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        avg_per_contract = df.groupby("Contract", observed=True)["MonthlyCharges_num"].mean().round(3).to_dict()
        metrics["avg_monthly_charges_per_contract"] = {k: (float(v) if pd.notna(v) else None) for k, v in avg_per_contract.items()}
    else:
        metrics["avg_monthly_charges_per_contract"] = {}

    # Count of tenure groups
    if "tenure_group" in df.columns:
        tenure_counts = df["tenure_group"].value_counts(dropna=False).to_dict()
        # convert numpy types
        metrics["tenure_group_counts"] = {str(k): int(v) for k, v in tenure_counts.items()}
    else:
        metrics["tenure_group_counts"] = {}

    # Internet service distribution
    if "InternetService" in df.columns:
        internet_dist = df["InternetService"].value_counts(dropna=False).to_dict()
        metrics["internet_service_distribution"] = {str(k): int(v) for k, v in internet_dist.items()}
    else:
        metrics["internet_service_distribution"] = {}

    # monthly charge segment churn rates (if monthly_charge_segment exists)
    if "monthly_charge_segment" in df.columns:
        seg = df.copy()
        seg["Churn"] = normalize_churn_values(seg)
        seg_grouped = seg.groupby("monthly_charge_segment", observed=True).agg(
            total=("Churn", "count"),
            churns=("Churn", lambda x: (x == "Yes").sum())
        )
        seg_grouped["churn_rate_pct"] = (100 * seg_grouped["churns"] / seg_grouped["total"]).round(3)
        metrics["churn_by_monthly_charge_segment"] = seg_grouped[["total", "churns", "churn_rate_pct"]].to_dict(orient="index")
    else:
        metrics["churn_by_monthly_charge_segment"] = {}

    return metrics


def pivot_churn_vs_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Return pivot table: Churn vs tenure_group (counts and churn rates)."""
    df2 = df.copy()
    df2["Churn_clean"] = normalize_churn_values(df2)
    # Count table
    count_tab = pd.crosstab(df2["tenure_group"], df2["Churn_clean"], dropna=False)
    # churn rate by tenure: percent of 'Yes' among each tenure_group
    if "Yes" in count_tab.columns:
        count_tab["churn_rate_pct"] = (100 * count_tab["Yes"] / count_tab.sum(axis=1)).round(3)
    else:
        count_tab["churn_rate_pct"] = 0.0
    # reset index for nicer CSV writing
    return count_tab.reset_index().rename_axis(None, axis=1)


def contract_counts(df: pd.DataFrame) -> pd.Series:
    if "Contract" in df.columns:
        return df["Contract"].value_counts(dropna=False)
    return pd.Series(dtype=int)


#  Save human-readable CSV summary 
def save_analysis_summary_csv(path: str, metrics: dict, pivot_df: pd.DataFrame, internet_dist: dict, contract_counts_ser: pd.Series):

    lines = []

    # Metrics: flattened
    lines.append(["METRIC", "VALUE"])
    # churn percentage + counts
    lines.append(["churn_percentage", metrics.get("churn_percentage")])
    lines.append(["churn_yes_count", metrics.get("churn_yes_count")])
    lines.append(["total_rows_counted_for_churn", metrics.get("total_rows_counted_for_churn")])

    # avg monthly charges per contract -> serialize as JSON-like string
    avg_per_contract = metrics.get("avg_monthly_charges_per_contract", {})
    lines.append(["avg_monthly_charges_per_contract", str(avg_per_contract)])

    # tenure group counts
    lines.append(["tenure_group_counts", str(metrics.get("tenure_group_counts", {}))])

    # internet distribution stored below too
    lines.append(["internet_service_distribution_summary", str(metrics.get("internet_service_distribution", {}))])

    # churn by monthly charge (if present)
    lines.append(["churn_by_monthly_charge_segment", str(metrics.get("churn_by_monthly_charge_segment", {}))])

    # Convert metrics block to DataFrame
    metrics_df = pd.DataFrame(lines[1:], columns=lines[0])

    # Prepare pivot and other tables
    pivot_block = pivot_df.copy()
    pivot_block.insert(0, "__section__", "PIVOT_Churn_vs_Tenure")

    # internet distribution DF
    internet_df = pd.DataFrame(list(internet_dist.items()), columns=["InternetService", "count"])
    internet_df.insert(0, "__section__", "InternetService_Distribution")

    # contract counts DF
    contract_df = contract_counts_ser.reset_index()
    contract_df.columns = ["Contract", "count"]
    contract_df.insert(0, "__section__", "Contract_Counts")

    # Now write all blocks sequentially to a single CSV
    # We'll write with a simple approach: write to CSV in append mode with headers for each block
    # Use a temporary DataFrame to write row-by-row
    with open(path, "w", encoding="utf-8", newline="") as f:
        # Metrics block
        metrics_df.to_csv(f, index=False)
        f.write("\n")

        # Pivot block
        pivot_block.to_csv(f, index=False)
        f.write("\n")

        # Internet block
        internet_df.to_csv(f, index=False)
        f.write("\n")

        # Contract block
        contract_df.to_csv(f, index=False)

    print("Saved analysis CSV to:", path)


#  Plotting 
def generate_plots(df: pd.DataFrame):
    """Generate optional plots and save to PLOTS_DIR"""
    df2 = df.copy()
    df2["Churn_clean"] = normalize_churn_values(df2)

    # 1) Churn rate by monthly charge segment (bar)
    if "monthly_charge_segment" in df2.columns:
        seg_tab = df2.groupby("monthly_charge_segment").agg(
            total=("Churn_clean", "count"),
            churns=("Churn_clean", lambda x: (x == "Yes").sum())
        )
        seg_tab["churn_rate_pct"] = 100 * seg_tab["churns"] / seg_tab["total"]
        seg_tab = seg_tab.sort_values("churn_rate_pct", ascending=False)

        plt.figure()
        seg_tab["churn_rate_pct"].plot(kind="bar")
        plt.title("Churn Rate by Monthly Charge Segment (%)")
        plt.ylabel("Churn Rate (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "churn_by_monthly_charge_segment.png"))
        plt.close()
        print("Saved plot:", os.path.join(PLOTS_DIR, "churn_by_monthly_charge_segment.png"))

    # 2) Histogram of TotalCharges
    if "TotalCharges" in df2.columns:
        plt.figure()
        vals = pd.to_numeric(df2["TotalCharges"], errors="coerce")
        vals.dropna(inplace=True)
        plt.hist(vals, bins=50)
        plt.title("Histogram of TotalCharges")
        plt.xlabel("TotalCharges")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "hist_total_charges.png"))
        plt.close()
        print("Saved plot:", os.path.join(PLOTS_DIR, "hist_total_charges.png"))

    # 3) Bar plot of Contract types
    if "Contract" in df2.columns:
        plt.figure()
        df2["Contract"].value_counts().plot(kind="bar")
        plt.title("Contract Type Counts")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "contract_type_counts.png"))
        plt.close()
        print("Saved plot:", os.path.join(PLOTS_DIR, "contract_type_counts.png"))


#  Main 
def main():
    print("\n🔎 Starting ETL Analysis\n")
    try:
        supabase = get_supabase_client()
    except Exception as e:
        print("❌ Supabase client error:", e)
        return

    # Fetch table
    try:
        df = fetch_table_from_supabase(supabase, TABLE_NAME)
    except Exception as e:
        print("❌ Error fetching data from Supabase:", e)
        return

    if df.empty:
        print("❌ No data fetched from Supabase. Exiting.")
        return

    # Clean up potential unnamed index column
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Ensure tenure_group and monthly_charge_segment exist; if not, create lightweight versions
    if "tenure_group" not in df.columns and "tenure" in df.columns:
        bins = [-1, 12, 36, 60, np.inf]
        labels = ["New", "Regular", "Loyal", "Champion"]
        df["tenure_group"] = pd.cut(pd.to_numeric(df["tenure"], errors="coerce"), bins=bins, labels=labels)

    if "monthly_charge_segment" not in df.columns and "MonthlyCharges" in df.columns:
        df["monthly_charge_segment"] = pd.cut(pd.to_numeric(df["MonthlyCharges"], errors="coerce"),
                                              bins=[0, 30, 70, np.inf], labels=["Low", "Medium", "High"])

    # Compute metrics & pivot
    metrics = compute_metrics(df)
    pivot = pivot_churn_vs_tenure(df)
    internet_dist = metrics.get("internet_service_distribution", {})
    contract_ser = contract_counts(df)

    # Save CSV summary
    save_analysis_summary_csv(SUMMARY_CSV_PATH, metrics, pivot, internet_dist, contract_ser)

    # Optional plots
    if GENERATE_PLOTS:
        generate_plots(df)

    print("\n✅ ETL Analysis complete.\n")


if __name__ == "__main__":
    main()
