import pandas as pd
import numpy as np
import re
from pathlib import Path

class MadridRentalELT:
    """
    ETL Pipeline for cleaning Madrid rental properties dataset
    
    Data Source: Kaggle - Madrid Housing Prices
    File: houses_madrid.csv
    
    Analysis: Rental properties (rent_price > 0)
    """
    
    def __init__(self, input_path='data/houses_madrid.csv'):
        """
        Initialize ELT Pipeline
        
        Args:
            input_path (str): Path to houses_madrid.csv from Kaggle
        """
        print(f"📂 Loading dataset from: {input_path}")
        self.df = pd.read_csv(input_path)
        self.df_clean = None
    
    def extract(self):
        """Extract: Filter properties with rent_price (rentals)"""
        print("\n" + "="*70)
        print("📥 EXTRACT: Filtering properties with rent_price...")
        print("="*70)
        
        print(f"   Initial records: {len(self.df):,}")
        
        print(f"\n   📊 Distribution of 'operation':")
        print(self.df['operation'].value_counts())
        
        print(f"\n   📊 Distribution of 'rent_price':")
        print(f"      NaN/Empty: {self.df['rent_price'].isnull().sum():,}")
        print(f"      With value: {self.df['rent_price'].notna().sum():,}")
        
        # KEY CHANGE: Filter by non-empty rent_price, not by operation
        print("\n   🔍 Filtering: non-empty rent_price...")
        self.df = self.df[self.df['rent_price'].notna()].copy()
        self.df = self.df[self.df['rent_price'] != ''].copy()
        
        print(f"\n✅ Rental records found: {len(self.df):,}")
        return self
    
    def transform(self):
        """Transform: Data cleaning and preprocessing"""
        print("\n" + "="*70)
        print("🔄 TRANSFORM: Processing data...")
        print("="*70)
        
        # 1. Convert rent_price to numeric (FIRST)
        print("\n   1️⃣  Converting rent_price to numeric...")
        self.df['rent_price'] = pd.to_numeric(self.df['rent_price'], errors='coerce')
        
        print(f"      Records with rent_price = NaN after conversion: {self.df['rent_price'].isnull().sum():,}")
        self.df = self.df[self.df['rent_price'] > 0].copy()
        print(f"      Final records with valid rent_price: {len(self.df):,}")
        
        if len(self.df) == 0:
            print("\n      ⚠️  WARNING: No valid rent_price values")
            return self
        
        # 2. Convert other numeric columns
        print("\n   2️⃣  Converting other numeric columns...")
        cols_num = ['rent_price_by_area', 'sq_mt_built', 'sq_mt_useful', 'built_year']
        
        for col in cols_num:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                print(f"      ✓ {col}: converted to float")
        
        # KEY CHANGE: n_rooms and n_bathrooms as INTEGER
        print("\n   2️⃣ b Converting n_rooms and n_bathrooms to INTEGER...")
        if 'n_rooms' in self.df.columns:
            self.df['n_rooms'] = pd.to_numeric(self.df['n_rooms'], errors='coerce')
            self.df['n_rooms'] = self.df['n_rooms'].fillna(0).astype(int)
            print(f"      ✓ n_rooms: converted to INT")
        
        if 'n_bathrooms' in self.df.columns:
            self.df['n_bathrooms'] = pd.to_numeric(self.df['n_bathrooms'], errors='coerce')
            self.df['n_bathrooms'] = self.df['n_bathrooms'].fillna(0).astype(int)
            print(f"      ✓ n_bathrooms: converted to INT")
        
        # 3. Extract district and neighborhood from neighborhood_id
        print("\n   3️⃣  Extracting district and neighborhood...")
        
        print(f"      Example of neighborhood_id:")
        if len(self.df) > 0:
            example = self.df['neighborhood_id'].iloc[0]
            print(f"      {example}")
        
        # Format: "Neighborhood 135: San Cristóbal (1308.89 €/m2) - District 21: Villaverde"
        self.df['neighborhood'] = self.df['neighborhood_id'].str.extract(r': (.*?) \(', expand=False)
        self.df['district'] = self.df['neighborhood_id'].str.extract(r'District \d+: (.*?)$', expand=False)
        
        print(f"      ✓ Neighborhoods extracted: {self.df['neighborhood'].nunique()} unique")
        print(f"      ✓ Districts extracted: {self.df['district'].nunique()} unique")
        
        # 4. Check null values BEFORE removing
        print("\n   4️⃣  Analyzing null values before cleaning...")
        critical_cols = ['rent_price', 'district', 'neighborhood', 'sq_mt_built', 'n_rooms', 'n_bathrooms']
        
        print("      Null values by critical column:")
        for col in critical_cols:
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / len(self.df) * 100) if len(self.df) > 0 else 0
            print(f"         {col:20s}: {null_count:6,} ({null_pct:5.1f}%)")
        
        # 5. Remove rows with critical null values
        print("\n   5️⃣  Removing rows with critical null values...")
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=critical_cols)
        eliminated = initial_count - len(self.df)
        
        print(f"      Records removed: {eliminated:,}")
        print(f"      Records remaining: {len(self.df):,}")
        
        if len(self.df) == 0:
            print("\n      ⚠️  WARNING: No records after removing null values")
            return self
        
        # 6. Remove price outliers (rational for rental)
        print("\n   6️⃣  Removing price outliers...")
        
        print(f"      rent_price statistics BEFORE cleaning outliers:")
        print(f"         Min: €{self.df['rent_price'].min():.2f}/month")
        print(f"         Max: €{self.df['rent_price'].max():.2f}/month")
        print(f"         Mean: €{self.df['rent_price'].mean():.2f}/month")
        print(f"         Median: €{self.df['rent_price'].median():.2f}/month")
        print(f"         P95: €{self.df['rent_price'].quantile(0.95):.2f}/month")
        
        # Remove outliers > P95 or > €5000/month
        p95 = self.df['rent_price'].quantile(0.95)
        max_price = max(5000, p95)
        
        outliers_count = len(self.df[self.df['rent_price'] > max_price])
        self.df = self.df[self.df['rent_price'] <= max_price]
        
        print(f"      Outliers limit: €{max_price:.2f}/month")
        print(f"      Outliers removed: {outliers_count:,}")
        print(f"      Final records: {len(self.df):,}")
        
        if len(self.df) == 0:
            print("\n      ⚠️  WARNING: No records after removing outliers")
            return self
        
        # 7. Calculate average price by district
        print("\n   7️⃣  Calculating statistics by district...")
        self.df['avg_district_rent'] = self.df.groupby('district')['rent_price'].transform('mean')
        self.df['median_district_rent'] = self.df.groupby('district')['rent_price'].transform('median')
        self.df['std_district_rent'] = self.df.groupby('district')['rent_price'].transform('std')
        
        # Calculate z-score (protect against division by zero)
        std_values = self.df.groupby('district')['rent_price'].transform('std')
        std_values = std_values.replace(0, 1)
        self.df['price_zscore'] = np.abs((self.df['rent_price'] - self.df['avg_district_rent']) / std_values)
        
        print(f"      ✓ Statistics calculated for {self.df['district'].nunique()} districts")
        
        # 8. Create target: overpriced (1=expensive, 0=fair)
        print("\n   8️⃣  Creating target variable (overpriced)...")
        self.df['overpriced'] = (self.df['rent_price'] > self.df['avg_district_rent']).astype(int)
        
        fair_price = (self.df['overpriced'] == 0).sum()
        expensive = (self.df['overpriced'] == 1).sum()
        total = len(self.df)
        
        print(f"      Fair price (0): {fair_price:,} ({fair_price/total*100:.1f}%)")
        print(f"      Expensive (1): {expensive:,} ({expensive/total*100:.1f}%)")
        
        # 9. Select final columns
        print("\n   9️⃣  Selecting relevant features...")
        feature_cols = [
            'id', 'rent_price', 'rent_price_by_area', 'sq_mt_built', 'sq_mt_useful', 
            'n_rooms', 'n_bathrooms', 'district', 'neighborhood',
            'avg_district_rent', 'median_district_rent', 'std_district_rent',
            'overpriced', 'built_year',
            'has_lift', 'has_ac', 'has_pool', 'has_parking', 'is_furnished'
        ]
        
        # Keep only existing columns
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        self.df_clean = self.df[feature_cols].copy()
        
        print(f"      ✓ {len(self.df_clean.columns)} features selected")
        
        return self
    
    def load(self, output_path='data/madrid_rent_clean.csv'):
        """Load: Save cleaned dataset"""
        print("\n" + "="*70)
        print(f"💾 LOAD: Saving to {output_path}...")
        print("="*70)
        
        # Validate data exists
        if self.df_clean is None or len(self.df_clean) == 0:
            print("\n❌ ERROR: No data to save")
            print("   Dataset is empty after cleaning")
            return None
        
        # Create directory if not exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df_clean.to_csv(output_path, index=False)
        
        print(f"✅ Cleaned dataset saved successfully")
        
        # Final statistics
        print("\n" + "="*70)
        print("📊 FINAL STATISTICS OF CLEANED DATASET")
        print("="*70)
        
        print(f"\n   📈 SIZE:")
        print(f"      Total records: {len(self.df_clean):,}")
        print(f"      Total columns: {len(self.df_clean.columns)}")
        
        print(f"\n   💰 RENTAL PRICES:")
        print(f"      Average: €{self.df_clean['rent_price'].mean():.2f}/month")
        print(f"      Median: €{self.df_clean['rent_price'].median():.2f}/month")
        print(f"      Min: €{self.df_clean['rent_price'].min():.2f}/month")
        print(f"      Max: €{self.df_clean['rent_price'].max():.2f}/month")
        print(f"      Std Dev: €{self.df_clean['rent_price'].std():.2f}")
        
        print(f"\n   📐 BUILT SQUARE METERS:")
        print(f"      Average: {self.df_clean['sq_mt_built'].mean():.1f} m²")
        print(f"      Median: {self.df_clean['sq_mt_built'].median():.1f} m²")
        print(f"      Min: {self.df_clean['sq_mt_built'].min():.1f} m²")
        print(f"      Max: {self.df_clean['sq_mt_built'].max():.1f} m²")
        
        print(f"\n   🏠 BEDROOMS:")
        print(f"      Average: {self.df_clean['n_rooms'].mean():.1f}")
        print(f"      Median: {self.df_clean['n_rooms'].median():.1f}")
        
        print(f"\n   🚿 BATHROOMS:")
        print(f"      Average: {self.df_clean['n_bathrooms'].mean():.1f}")
        print(f"      Median: {self.df_clean['n_bathrooms'].median():.1f}")
        
        print(f"\n   🏘️  GEOGRAPHY:")
        print(f"      Unique districts: {self.df_clean['district'].nunique()}")
        print(f"      Unique neighborhoods: {self.df_clean['neighborhood'].nunique()}")
        
        print(f"\n   🎯 TARGET (overpriced):")
        fair_price = (self.df_clean['overpriced'] == 0).sum()
        expensive = (self.df_clean['overpriced'] == 1).sum()
        total = len(self.df_clean)
        
        print(f"      Fair price (0): {fair_price:,} ({fair_price/total*100:.1f}%)")
        print(f"      Expensive (1): {expensive:,} ({expensive/total*100:.1f}%)")
        
        print(f"\n   💼 FEATURES ({len(self.df_clean.columns)} columns):")
        print(f"      Data types:")
        for col in self.df_clean.columns:
            dtype = self.df_clean[col].dtype
            print(f"         {col:25s}: {dtype}")
        
        print("\n✅ ELT COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return self.df_clean


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """
    Run the complete ELT pipeline
    
    Usage:
        python src/etl.py
    """
    
    print("\n🚀 Starting ETL Pipeline for Madrid Rental Prices")
    print("=" * 70)
    print("Data Source: Kaggle")
    print("Dataset: Madrid Housing Prices (houses_madrid.csv)")
    print("Analysis: Rental Properties (rent_price > 0)")
    print("=" * 70)
    
    try:
        elt = MadridRentalELT('data/houses_madrid.csv')
        df_clean = elt.extract().transform().load('data/madrid_rent_clean.csv')
        
        if df_clean is not None and len(df_clean) > 0:
            print("\n🎉 Pipeline completed successfully!")
        else:
            print("\n⚠️  Pipeline completed but no data to save")
        
    except FileNotFoundError:
        print("\n❌ ERROR: File 'data/houses_madrid.csv' not found")
        print("   Download dataset from Kaggle and place it in 'data/' folder")
    except Exception as e:
        print(f"\n❌ ERROR during ETL: {str(e)}")
        import traceback
        traceback.print_exc()