import pandas as pd
import os
import numpy as np

def transform_data(raw_path):
    base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    staged_dir=os.path.join(base_dir,'data','staged')
    os.makedirs(staged_dir,exist_ok=True)

    df=pd.read_csv(raw_path)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df['TotalCharges']=df['TotalCharges'].fillna(df['TotalCharges'].median())

    bins = [-1, 12, 36, 60, np.inf]
    labels = ["New", "Regular", "Loyal", "Champion"]
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=True)
    
    df['monthly_charge_segment']=pd.cut(df['MonthlyCharges'],bins=[0,30,70,np.inf],labels=['Low','Medium','High'])

    df['has_internet_service']=(df['InternetService']!='No').astype(int)

    df['is_multi_line_user']=(df['MultipleLines']!='No').astype(int)

    df['contract_type_code']=df['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})

    df.drop(columns=['customerID','gender'],inplace=True,errors='ignore')

    staged_path=os.path.join(staged_dir,'churn_transformed.csv')
    df.to_csv(staged_path,index=False)

    print(f"✅ Data transformed and saved at: {staged_path}")
    return staged_path


if __name__=='__main__':
    from extract import extract_data
    raw_path=extract_data()
    transform_data(raw_path)