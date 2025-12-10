import os
import pandas as pd

def extract_data():
    base_dir=os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    data_dir=os.path.join(base_dir,'data','raw')
    os.makedirs(data_dir,exist_ok=True)

    # Path where input CSV must be located
    input_csv = os.path.join(base_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Read the CSV
    df = pd.read_csv(input_csv)

    # Output path
    raw_path = os.path.join(data_dir, 'Telco_Customer_raw.csv')


    # Save cleaned CSV
    df.to_csv(raw_path, index=False)

    print(f"✅ Data extracted and saved at: {raw_path}")
    return raw_path
 
if __name__ == "__main__":
    extract_data()