import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def validate_data():
    print("\n Starting Validation...\n")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    staged_path = os.path.join(base_dir, "data", "staged", "churn_transformed.csv")

    if not os.path.exists(staged_path):
        print(f"❌ ERROR: File not found: {staged_path}")
        return

    df = pd.read_csv(staged_path)
    supabase = get_supabase()

    # --- Fetch row count from Supabase ---
    try:
        count_response = supabase.table("telco_data").select("id", count="exact").limit(1).execute()
        supabase_count = count_response.count
    except Exception as e:
        print(f"❌ ERROR fetching Supabase count: {e}")
        return

    # -------------------------
    # VALIDATIONS
    # -------------------------
    results = {}

    # Check missing values
    required_no_null = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in required_no_null:
        missing = df[col].isna().sum()
        results[f"Missing values in '{col}'"] = (missing == 0, missing)

    # Unique row count (should match df count)
    results["Unique row count matches dataset"] = (df.drop_duplicates().shape[0] == df.shape[0], None)

    # Supabase count matches CSV count
    results["Supabase row count matches CSV"] = (supabase_count == df.shape[0], (supabase_count, df.shape[0]))

    # All tenure_group segments exist
    expected_tenure_groups = {"New", "Regular", "Loyal", "Champion"}
    actual_tenure_groups = set(df["tenure_group"].unique())
    results["All tenure_group categories exist"] = (expected_tenure_groups.issubset(actual_tenure_groups), actual_tenure_groups)

    # All monthly_charge_segment exist
    expected_charge_groups = {"Low", "Medium", "High"}
    actual_charge_groups = set(df["monthly_charge_segment"].unique())
    results["All monthly_charge_segment categories exist"] = (expected_charge_groups.issubset(actual_charge_groups), actual_charge_groups)

    # Contract codes check
    valid_contract_codes = {0, 1, 2}
    actual_codes = set(df["contract_type_code"].unique())
    results["Contract codes are valid (0,1,2)"] = (actual_codes.issubset(valid_contract_codes), actual_codes)

    # -------------------------
    # PRINT SUMMARY
    # -------------------------
    print("\n📋 VALIDATION SUMMARY\n")

    for check, (passed, details) in results.items():
        icon = "✅" if passed else "❌"
        print(f"{icon} {check}", end="")

        if details is not None:
            print(f" → {details}")
        else:
            print()

    print("\n🎯 Validation Completed.\n")


if __name__ == "__main__":
    validate_data()
