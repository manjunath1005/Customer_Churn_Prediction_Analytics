import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

# ------------------------------------------------------
# Initialize Supabase client
# ------------------------------------------------------
def get_supabase_client():
    """Initialize and return Supabase client."""
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_KEY in .env")

    return create_client(url, key)


# ------------------------------------------------------
# Step 2: Load CSV data into Supabase table
# ------------------------------------------------------
def load_to_supabase(staged_path: str, table_name: str = "telco_data"):
    """
    Load transformed CSV into a Supabase table.

    Args:
        staged_path (str): Path to the transformed CSV file.
        table_name (str): Supabase table name.
    """

    # Build absolute path
    if not os.path.isabs(staged_path):
        staged_path = os.path.abspath(os.path.join(os.path.dirname(__file__), staged_path))

    print(f"🔍 Looking for data file at: {staged_path}")

    if not os.path.exists(staged_path):
        print(f"❌ Error: File not found at {staged_path}")
        print("ℹ️  Please run transform.py first.")
        return

    try:
        supabase = get_supabase_client()

        df = pd.read_csv(staged_path)
        total_rows = len(df)
        batch_size = 250  # Adjust batch size as needed

        print(f"📊 Loading {total_rows} rows into '{table_name}'...")

        # Insert in batches
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i + batch_size].copy()

            # Convert NaN → None for PostgreSQL compatibility
            batch = batch.where(pd.notnull(batch), None)
            records = batch.to_dict("records")

            try:
                response = supabase.table(table_name).insert(records).execute()
                if hasattr(response, 'error') and response.error:
                    print(f"⚠️  Batch {i//batch_size+1} error: {response.error}")
                else:
                    print(f"✅ Inserted rows {i+1}-{min(i+batch_size, total_rows)}")
            except Exception as e:
                print(f"⚠️  Error inserting batch {i//batch_size+1}: {str(e)}")

        print(f"🎯 Finished loading data into '{table_name}'.")
    except Exception as e:
        print(f"❌ Error loading data: {e}")


# ------------------------------------------------------
# Step 3: Run as standalone script
# ------------------------------------------------------
if __name__ == "__main__":
    staged_csv_path = os.path.join("..", "data", "staged", "churn_transformed.csv")
    load_to_supabase(staged_csv_path)
