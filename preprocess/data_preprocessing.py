
import pandas as pd
import sqlite3
import os
import numpy as np
from datetime import datetime

def load_and_clean_dataset1(filepath: str) -> pd.DataFrame:
    
    
    # Load data
    df = pd.read_excel(filepath)
    print(f"\n Loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    
    initial_rows = len(df)
    
    # DATA CLEANING STEPS
    print("\n DATA CLEANING PROCESS:")
    print("-" * 70)
    
    # Step 1: Check for missing values
    print("\n 1️.  Checking for missing values...")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("   Missing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"      {col}: {count} ({count/len(df)*100:.1f}%)")
        
       
        print("\n   Handling Pregnancy column...")
        males_before = len(df[df['Sex'] == 0])
        df.loc[df['Sex'] == 0, 'Pregnancy'] = 0  # Males: Pregnancy = 0
        print(f"      Set Pregnancy=0 for {males_before} males")
        
        # For females with missing pregnancy, assume not pregnant (0)
        females_missing_count = df[(df['Sex'] == 1) & (df['Pregnancy'].isnull())].shape[0]
        if females_missing_count > 0:
            df.loc[(df['Sex'] == 1) & (df['Pregnancy'].isnull()), 'Pregnancy'] = 0
            print(f"      Set Pregnancy=0 for {females_missing_count} females with missing values (assumed not pregnant)")
        
        # Handle alcohol_consumption_per_day: Missing = non-drinker (0)
        alcohol_missing = df['alcohol_consumption_per_day'].isnull().sum()
        if alcohol_missing > 0:
            df['alcohol_consumption_per_day'] = df['alcohol_consumption_per_day'].fillna(0)
            print(f"      Filled {alcohol_missing} missing alcohol values with 0 (non-drinkers)")
        
        # Handle Genetic_Pedigree_Coefficient: Impute with median instead of dropping
        genetic_missing = df['Genetic_Pedigree_Coefficient'].isnull().sum()
        if genetic_missing > 0:
            median_val = df['Genetic_Pedigree_Coefficient'].median()
            df['Genetic_Pedigree_Coefficient'] = df['Genetic_Pedigree_Coefficient'].fillna(median_val)
            print(f"      Filled {genetic_missing} missing Genetic coefficient with median ({median_val:.4f})")
        
        print(f"\n   Smart cleaning complete: {initial_rows - len(df)} rows removed, {len(df)} retained ({len(df)/initial_rows*100:.1f}%)")
    else:
        print("   No missing values found")
    
    # Step 2: Remove duplicates
    print("\n2️.  Checking for duplicate Patient_Numbers...")
    duplicates = df.duplicated(subset=['Patient_Number'], keep='first')
    if duplicates.sum() > 0:
        df = df[~duplicates]
        print(f"   Removed {duplicates.sum()} duplicate patients")
    else:
        print("   No duplicates found")
    
    # Step 3: Validate binary columns (0 or 1)
    print("\n3️.  Validating binary columns (must be 0 or 1)...")
    binary_cols = [
        'Blood_Pressure_Abnormality',
        'Sex',
        'Pregnancy',
        'Smoking',
        'Chronic_kidney_disease',
        'Adrenal_and_thyroid_disorders'
    ]
    
    for col in binary_cols:
        if col in df.columns:
            invalid = ~df[col].isin([0, 1])
            if invalid.sum() > 0:
                print(f"     {col}: Removed {invalid.sum()} invalid values")
                df = df[~invalid]
            else:
                print(f"    {col}: Valid")
    
    # Step 4: Validate Stress Level (1, 2, or 3)
    print("\n4️.  Validating Stress Level (must be 1, 2, or 3)...")
    if 'Level_of_Stress' in df.columns:
        invalid_stress = ~df['Level_of_Stress'].isin([1, 2, 3])
        if invalid_stress.sum() > 0:
            df = df[~invalid_stress]
            print(f"     Removed {invalid_stress.sum()} rows with invalid stress levels")
        else:
            print("    All stress levels valid")
    
    # Step 5: Validate numeric ranges
    print("\n5️.  Validating numeric ranges...")
    
    validations = {
        'Age': (0, 120, "years"),
        'BMI': (10, 60, "kg/m²"),
        'Level_of_Hemoglobin': (0, 25, "g/dl"),
        'Genetic_Pedigree_Coefficient': (0, 1, "ratio"),
        'salt_content_in_the_diet': (0, 60000, "mg/day"),  # Adjusted: data median ~25,000
        'alcohol_consumption_per_day': (0, 500, "ml/day")
    }
    
    for col, (min_val, max_val, unit) in validations.items():
        if col in df.columns:
            invalid = (df[col] < min_val) | (df[col] > max_val)
            if invalid.sum() > 0:
                print(f"     {col}: Removed {invalid.sum()} outliers (valid: {min_val}-{max_val} {unit})")
                df = df[~invalid]
            else:
                print(f"    {col}: All values in range {min_val}-{max_val} {unit}")
    
    # Step 6: Fix data types
    print("\n6️.  Ensuring correct data types...")
    df['Patient_Number'] = df['Patient_Number'].astype(int)
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    if 'Level_of_Stress' in df.columns:
        df['Level_of_Stress'] = df['Level_of_Stress'].astype(int)
    print("    Data types corrected")
    
    # Summary
    print(f"\n📊 CLEANING SUMMARY:")
    print(f"   Initial rows: {initial_rows}")
    print(f"   Final rows: {len(df)}")
    print(f"   Removed: {initial_rows - len(df)} ({(initial_rows - len(df))/initial_rows*100:.1f}%)")
    print(f"   Data quality: {len(df)/initial_rows*100:.1f}%")
    
    return df


def load_and_clean_dataset2(filepath: str) -> pd.DataFrame:
    
    
    
    
    # Load data
    df = pd.read_excel(filepath)
    print(f"\n✅ Loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    
    initial_rows = len(df)
    
    # DATA CLEANING
    print("\n DATA CLEANING PROCESS:")
    print("-" * 70)
    
    # Step 1: Missing values
    print("\n1️. Checking for missing values...")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("   Missing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"      {col}: {count} ({count/len(df)*100:.1f}%)")
        
        # Impute missing Physical_activity with patient mean, fallback to global median
        pa_missing = df['Physical_activity'].isnull().sum()
        if pa_missing > 0:
            # Calculate per-patient mean (from non-null days)
            patient_means = df.groupby('Patient_Number')['Physical_activity'].transform('mean')
            df['Physical_activity'] = df['Physical_activity'].fillna(patient_means)
            
            # Fallback: if a patient has ALL days missing, use global median
            still_missing = df['Physical_activity'].isnull().sum()
            if still_missing > 0:
                global_median = df['Physical_activity'].median()
                df['Physical_activity'] = df['Physical_activity'].fillna(global_median)
                print(f"      Filled {still_missing} remaining missing values with global median ({global_median:.0f})")
            
            print(f"    Imputed {pa_missing} missing Physical_activity values (patient mean + global median fallback)")
    else:
        print("    No missing values found")
    
    # Step 2: Validate Day_Number (1-10)
    print("\n2️.  Validating Day_Number (must be 1-10)...")
    if 'Day_Number' in df.columns:
        invalid_days = ~df['Day_Number'].between(1, 10)
        if invalid_days.sum() > 0:
            df = df[~invalid_days]
            print(f"     Removed {invalid_days.sum()} rows with invalid day numbers")
        else:
            print("    All day numbers valid (1-10)")
    
    # Step 3: Validate Physical_activity
    print("\n3️.  Validating Physical_activity (steps)...")
    if 'Physical_activity' in df.columns:
        # Negative steps
        negative = df['Physical_activity'] < 0
        if negative.sum() > 0:
            df = df[~negative]
            print(f"     Removed {negative.sum()} rows with negative steps")
        
        # Unrealistic high values (>50,000 steps)
        # NOTE: Your data has high step counts (avg ~25,000). Adjust threshold if needed.
        too_high = df['Physical_activity'] > 50000
        if too_high.sum() > 0:
            df = df[~too_high]
            print(f"     Removed {too_high.sum()} rows with unrealistic steps (>50,000)")
        
        if negative.sum() == 0 and too_high.sum() == 0:
            print("    All physical activity values valid")
            # Show statistics for transparency
            print(f"      Range: {df['Physical_activity'].min():.0f} - {df['Physical_activity'].max():.0f} steps")
            print(f"      Mean: {df['Physical_activity'].mean():.0f} steps/day")
    
    # Step 4: Ensure data types
    print("\n4️.  Ensuring correct data types...")
    df['Patient_Number'] = df['Patient_Number'].astype(int)
    df['Day_Number'] = df['Day_Number'].astype(int)
    df['Physical_activity'] = df['Physical_activity'].astype(int)
    print("    Data types corrected")
    
    # Summary
    print(f"\n CLEANING SUMMARY:")
    print(f"   Initial rows: {initial_rows}")
    print(f"   Final rows: {len(df)}")
    print(f"   Removed: {initial_rows - len(df)} ({(initial_rows - len(df))/initial_rows*100:.1f}%)")
    
    return df


def generate_data_quality_report(df1: pd.DataFrame, df2: pd.DataFrame):
    
    
    print("\n" + "="*70)
    print(" DATA QUALITY & SUMMARY REPORT")
    print("="*70)
    
    # DATASET 1 STATISTICS
    print("\n DATASET 1 - PATIENT HEALTH METRICS")
    print("-" * 70)
    
    print(f"\n Sample Size:")
    print(f"   Total Patients: {len(df1):,}")
    
    print(f"\n Demographics:")
    male_count = len(df1[df1['Sex'] == 0])
    female_count = len(df1[df1['Sex'] == 1])
    print(f"   Male: {male_count:,} ({male_count/len(df1)*100:.1f}%)")
    print(f"   Female: {female_count:,} ({female_count/len(df1)*100:.1f}%)")
    
    print(f"\n Age Distribution:")
    print(f"   Min: {df1['Age'].min():.0f} years")
    print(f"   Max: {df1['Age'].max():.0f} years")
    print(f"   Mean: {df1['Age'].mean():.1f} years")
    print(f"   Median: {df1['Age'].median():.0f} years")
    print(f"   Std Dev: {df1['Age'].std():.1f} years")
    
    print(f"\n BMI Distribution:")
    print(f"   Min: {df1['BMI'].min():.1f}")
    print(f"   Max: {df1['BMI'].max():.1f}")
    print(f"   Mean: {df1['BMI'].mean():.1f}")
    print(f"   Median: {df1['BMI'].median():.1f}")
    
    # BMI Categories
    underweight = len(df1[df1['BMI'] < 18.5])
    normal = len(df1[(df1['BMI'] >= 18.5) & (df1['BMI'] < 25)])
    overweight = len(df1[(df1['BMI'] >= 25) & (df1['BMI'] < 30)])
    obese = len(df1[df1['BMI'] >= 30])
    
    print(f"   Categories:")
    print(f"      Underweight (<18.5): {underweight} ({underweight/len(df1)*100:.1f}%)")
    print(f"      Normal (18.5-24.9): {normal} ({normal/len(df1)*100:.1f}%)")
    print(f"      Overweight (25-29.9): {overweight} ({overweight/len(df1)*100:.1f}%)")
    print(f"      Obese (≥30): {obese} ({obese/len(df1)*100:.1f}%)")
    
    print(f"\n Lifestyle Factors:")
    smokers = len(df1[df1['Smoking'] == 1])
    print(f"   Smokers: {smokers:,} ({smokers/len(df1)*100:.1f}%)")
    print(f"   Average Alcohol Consumption: {df1['alcohol_consumption_per_day'].mean():.1f} ml/day")
    print(f"   Average Salt Intake: {df1['salt_content_in_the_diet'].mean():.1f} mg/day")
    
    print(f"\n Stress Levels:")
    for level, label in [(1, 'Low'), (2, 'Normal'), (3, 'High')]:
        count = len(df1[df1['Level_of_Stress'] == level])
        print(f"   {label}: {count:,} ({count/len(df1)*100:.1f}%)")
    
    print(f"\n Health Conditions:")
    bp_abnormal = len(df1[df1['Blood_Pressure_Abnormality'] == 1])
    kidney = len(df1[df1['Chronic_kidney_disease'] == 1])
    thyroid = len(df1[df1['Adrenal_and_thyroid_disorders'] == 1])
    
    print(f"   Abnormal Blood Pressure: {bp_abnormal:,} ({bp_abnormal/len(df1)*100:.1f}%)")
    print(f"   Chronic Kidney Disease: {kidney:,} ({kidney/len(df1)*100:.1f}%)")
    print(f"   Thyroid/Adrenal Disorders: {thyroid:,} ({thyroid/len(df1)*100:.1f}%)")
    
    print(f"\n Hemoglobin Levels:")
    print(f"   Mean: {df1['Level_of_Hemoglobin'].mean():.2f} g/dl")
    print(f"   Range: {df1['Level_of_Hemoglobin'].min():.1f} - {df1['Level_of_Hemoglobin'].max():.1f} g/dl")
    
    # DATASET 2 STATISTICS
    print("\n" + "-" * 70)
    print(" DATASET 2 - PHYSICAL ACTIVITY TRACKING")
    print("-" * 70)
    
    print(f"\n Sample Size:")
    print(f"   Total Records: {len(df2):,}")
    print(f"   Unique Patients: {df2['Patient_Number'].nunique():,}")
    print(f"   Days Tracked: {df2['Day_Number'].nunique()}")
    
    print(f"\n Physical Activity Statistics:")
    print(f"   Mean Daily Steps: {df2['Physical_activity'].mean():.0f}")
    print(f"   Median Daily Steps: {df2['Physical_activity'].median():.0f}")
    print(f"   Min Steps: {df2['Physical_activity'].min():,}")
    print(f"   Max Steps: {df2['Physical_activity'].max():,}")
    print(f"   Std Dev: {df2['Physical_activity'].std():.0f}")
    
    # Activity Categories
    sedentary = len(df2[df2['Physical_activity'] < 5000])
    low_active = len(df2[(df2['Physical_activity'] >= 5000) & (df2['Physical_activity'] < 7500)])
    active = len(df2[(df2['Physical_activity'] >= 7500) & (df2['Physical_activity'] < 10000)])
    very_active = len(df2[df2['Physical_activity'] >= 10000])
    
    print(f"\n Activity Categories:")
    print(f"   Sedentary (<5,000 steps): {sedentary:,} ({sedentary/len(df2)*100:.1f}%)")
    print(f"   Low Active (5,000-7,499): {low_active:,} ({low_active/len(df2)*100:.1f}%)")
    print(f"   Active (7,500-9,999): {active:,} ({active/len(df2)*100:.1f}%)")
    print(f"   Very Active (≥10,000): {very_active:,} ({very_active/len(df2)*100:.1f}%)")
    
    # DATA INTEGRITY CHECK
    print("\n" + "-" * 70)
    print(" DATA INTEGRITY VALIDATION")
    print("-" * 70)
    
    # Check for orphaned records
    patients_in_df1 = set(df1['Patient_Number'])
    patients_in_df2 = set(df2['Patient_Number'])
    
    orphaned = patients_in_df2 - patients_in_df1
    missing = patients_in_df1 - patients_in_df2
    
    print(f"\n Cross-Dataset Validation:")
    print(f"   Patients in Dataset 1: {len(patients_in_df1):,}")
    print(f"   Patients in Dataset 2: {len(patients_in_df2):,}")
    
    if orphaned:
        print(f" WARNING: {len(orphaned)} patients in Dataset 2 have no health records")
    else:
        print(f" All Dataset 2 patients have health records")
    
    if missing:
        print(f"INFO: {len(missing)} patients have no activity tracking")
    else:
        print(f"All patients have activity tracking")
    
    print("\n" + "="*70)


def create_database(df1: pd.DataFrame, df2: pd.DataFrame, db_path: str):
    """
    Create SQLite database with cleaned data and indexes
    """
    
    print("\n" + "="*70)
    print("CREATING SQLITE DATABASE")
    print("="*70)
    
    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"\n🗑️  Removed existing database: {db_path}")
    
    # Create new database
    print(f"\nCreating new database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # Load tables
    print("\nLoading tables...")
    df1.to_sql('health_dataset_1', conn, if_exists='replace', index=False)
    print(f"   Created table: health_dataset_1 ({len(df1):,} rows)")
    
    df2.to_sql('health_dataset_2', conn, if_exists='replace', index=False)
    print(f"   Created table: health_dataset_2 ({len(df2):,} rows)")
    
    # Create indexes for performance
    print("\n🔧 Creating indexes for query optimization...")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_patient_1 
        ON health_dataset_1(Patient_Number)
    """)
    print("   Created index: idx_patient_1")
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_patient_2 
        ON health_dataset_2(Patient_Number)
    """)
    print("Created index: idx_patient_2")
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_day 
        ON health_dataset_2(Day_Number)
    """)
    print(" Created index: idx_day")
    
    conn.commit()
    
    # Verify
    print("\nVerifying database integrity...")
    cursor.execute("SELECT COUNT(*) FROM health_dataset_1")
    count1 = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM health_dataset_2")
    count2 = cursor.fetchone()[0]
    
    print(f"   health_dataset_1: {count1:,} rows")
    print(f"   health_dataset_2: {count2:,} rows")
    
    conn.close()
    
    print(f"\nDatabase created successfully!")
    print("="*70)


def setup_database(
    excel_file1=None,
    excel_file2=None,
    db_path=None
):
    
    
    # Get project root directory (parent of scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Set default paths relative to project root
    if excel_file1 is None:
        excel_file1 = os.path.join(project_root, "data", "raw", "dataset1.xlsm")
    if excel_file2 is None:
        excel_file2 = os.path.join(project_root, "data", "raw", "dataset2.xlsm")
    if db_path is None:
        db_path = os.path.join(project_root, "data", "health_data.db")
    
    print("\n" + "="*70)
    print("HEALTH DATA PREPROCESSING PIPELINE")
    print("="*70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Load and clean Dataset 1
        df1 = load_and_clean_dataset1(excel_file1)
        
        # Step 2: Load and clean Dataset 2
        df2 = load_and_clean_dataset2(excel_file2)
        
        # Step 3: Generate data quality report
        generate_data_quality_report(df1, df2)
        
        # Step 4: Create database
        create_database(df1, df2, db_path)
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE!")
        print("="*70)
        print(f"\nDatabase ready: {db_path}")
        print("Total patients: {:,}".format(len(df1)))
        print("Total activity records: {:,}".format(len(df2)))
        print("\nYou can now run the main analysis pipeline!")
        print("="*70 + "\n")
        
        return db_path
        
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find Excel files")
        print(f"   Please verify the file paths:")
        print(f"   - {excel_file1}")
        print(f"   - {excel_file2}")
        raise
        
    except Exception as e:
        print(f"\nERROR during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    setup_database()