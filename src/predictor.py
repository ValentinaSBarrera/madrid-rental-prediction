import pandas as pd
import numpy as np
from model import RentalPriceModel
from etl import MadridRentalELT

class RentalPredictor:
    """Orchestrator: ETL → Model → Prediction"""
    
    def __init__(self, csv_path='data/houses_madrid.csv'):
        self.csv_path = csv_path
        self.model = RentalPriceModel()
        self.df_clean = None
        self.district_stats = None
    
    def run_etl(self, output_path='data/madrid_rent_clean.csv'):
        """Execute complete ETL pipeline"""
        print("\n" + "="*70)
        print("🔄 STARTING ETL PIPELINE")
        print("="*70)
        
        elt = MadridRentalELT(self.csv_path)
        self.df_clean = elt.extract().transform().load(output_path)
        
        # Save statistics by district
        self.district_stats = self.df_clean.groupby('district').agg({
            'rent_price': ['mean', 'median', 'std', 'count'],
            'sq_mt_built': 'mean',
            'n_rooms': 'mean'
        }).round(2)
        
        return self.df_clean
    
    def train_model(self, df=None):
        """Train ML model"""
        print("\n" + "="*70)
        print("🎓 TRAINING MODEL")
        print("="*70)
        
        if df is None:
            df = self.df_clean
        
        if df is None:
            raise ValueError("Dataset not loaded. Run run_etl() first")
        
        self.model.train(df)
        self.model.save('model/rental_model.pkl')
        
        return self.model
    
    def predict_single(self, sq_mt_built, sq_mt_useful, n_rooms, n_bathrooms, district):
        """Predict ONE property"""
        print(f"\n🔍 PREDICTION FOR:")
        print(f"   District: {district}")
        print(f"   Square meters: {sq_mt_built} m² | Bedrooms: {n_rooms} | Bathrooms: {n_bathrooms}")
        
        result, error = self.model.predict(sq_mt_built, sq_mt_useful, n_rooms, n_bathrooms, district)
        
        if error:
            print(f"❌ Error: {error}")
            return None
        
        return result
    
    def predict_batch(self, csv_path):
        """Predict BATCH of properties"""
        print(f"\n📊 BATCH PREDICTION from {csv_path}")
        
        df_batch = pd.read_csv(csv_path)
        predictions = []
        
        for idx, row in df_batch.iterrows():
            result, error = self.model.predict(
                sq_mt_built=row['sq_mt_built'],
                sq_mt_useful=row['sq_mt_useful'],
                n_rooms=row['n_rooms'],
                n_bathrooms=row['n_bathrooms'],
                district=row['district']
            )
            
            if error is None:
                predictions.append({
                    'id': idx,
                    'district': row['district'],
                    'overpriced': result['overpriced'],
                    'label': result['label'],
                    'confidence': result['confidence_pct']
                })
        
        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_csv('data/batch_predictions.csv', index=False)
        
        print(f"\n✅ {len(predictions)}/{len(df_batch)} predictions completed")
        print(f"💾 Saved to: data/batch_predictions.csv")
        print(f"\nSummary:")
        print(df_predictions['label'].value_counts())
        
        return df_predictions
    
    def get_district_analysis(self, district):
        """Detailed analysis of a district"""
        print(f"\n📍 ANALYSIS OF DISTRICT: {district}")
        print("="*70)
        
        if self.df_clean is None:
            raise ValueError("Dataset not loaded. Run run_etl() first")
        
        df_district = self.df_clean[self.df_clean['district'] == district]
        
        if len(df_district) == 0:
            print(f"❌ District not found")
            return None
        
        stats = {
            'total_props': len(df_district),
            'avg_price': df_district['rent_price'].mean(),
            'median_price': df_district['rent_price'].median(),
            'std_price': df_district['rent_price'].std(),
            'min_price': df_district['rent_price'].min(),
            'max_price': df_district['rent_price'].max(),
            'avg_sqm': df_district['sq_mt_built'].mean(),
            'price_per_sqm': (df_district['rent_price'] / df_district['sq_mt_built']).mean(),
            'pct_expensive': (df_district['overpriced'].sum() / len(df_district) * 100),
            'neighborhoods': df_district['neighborhood'].nunique()
        }
        
        print(f"   📊 Total properties: {stats['total_props']}")
        print(f"   💰 Average price: €{stats['avg_price']:.2f}/month")
        print(f"   📈 Median price: €{stats['median_price']:.2f}/month")
        print(f"   📉 Standard deviation: €{stats['std_price']:.2f}")
        print(f"   💵 Range: €{stats['min_price']:.2f} - €{stats['max_price']:.2f}")
        print(f"   📐 Average m²: {stats['avg_sqm']:.2f} m²")
        print(f"   🔢 €/m²: €{stats['price_per_sqm']:.2f}")
        print(f"   🔴 % Expensive: {stats['pct_expensive']:.1f}%")
        print(f"   🏘️  Unique neighborhoods: {stats['neighborhoods']}")
        
        return stats
    
    def compare_districts(self, districts=None):
        """Compare multiple districts"""
        print("\n📊 DISTRICT COMPARISON")
        print("="*70)
        
        if self.df_clean is None:
            raise ValueError("Dataset not loaded. Run run_etl() first")
        
        if districts is None:
            # Top 10 districts by property count
            districts = self.df_clean['district'].value_counts().head(10).index.tolist()
        
        comparison = []
        
        for district in districts:
            df_d = self.df_clean[self.df_clean['district'] == district]
            comparison.append({
                'District': district,
                'Properties': len(df_d),
                'Avg Price (€)': f"{df_d['rent_price'].mean():.0f}",
                'Median Price (€)': f"{df_d['rent_price'].median():.0f}",
                '€/m²': f"{(df_d['rent_price'] / df_d['sq_mt_built']).mean():.1f}",
                '% Expensive': f"{(df_d['overpriced'].sum() / len(df_d) * 100):.1f}%"
            })
        
        df_comparison = pd.DataFrame(comparison)
        print(df_comparison.to_string(index=False))
        df_comparison.to_csv('data/district_comparison.csv', index=False)
        print(f"\n💾 Saved to: data/district_comparison.csv")
        
        return df_comparison
    
    def recommend_neighborhoods(self, budget, n_rooms, n_bathrooms, top_n=10):
        """Recommend neighborhoods by budget"""
        print(f"\n🎯 NEIGHBORHOOD RECOMMENDATIONS")
        print(f"   Budget: €{budget}/month")
        print(f"   Bedrooms: {n_rooms} | Bathrooms: {n_bathrooms}")
        print("="*70)
        
        if self.df_clean is None:
            raise ValueError("Dataset not loaded. Run run_etl() first")
        
        # Filter by specs
        df_filtered = self.df_clean[
            (self.df_clean['rent_price'] <= budget) &
            (self.df_clean['n_rooms'] >= n_rooms) &
            (self.df_clean['n_bathrooms'] >= n_bathrooms)
        ]
        
        if len(df_filtered) == 0:
            print(f"❌ No neighborhoods meet the criteria")
            return None
        
        # Group by neighborhood
        recommendations = df_filtered.groupby('neighborhood').agg({
            'rent_price': ['mean', 'count'],
            'district': 'first',
            'overpriced': lambda x: (x.sum() / len(x) * 100)
        }).round(2)
        
        recommendations.columns = ['Avg Price', 'Properties', 'District', '% Expensive']
        recommendations = recommendations.sort_values('Avg Price')
        recommendations = recommendations.head(top_n)
        
        print(recommendations.to_string())
        
        return recommendations
    
    def full_pipeline(self):
        """Execute COMPLETE pipeline"""
        print("\n" + "🚀"*30)
        print("🚀 COMPLETE PIPELINE: ETL → MODEL → PREDICTION")
        print("🚀"*30)
        
        # 1. ETL
        self.run_etl()
        
        # 2. Model
        self.train_model()
        
        # 3. Analysis
        print("\n" + "="*70)
        print("📊 GENERAL ANALYSIS")
        print("="*70)
        
        self.compare_districts()
        
        # 4. Recommendations
        print("\n" + "="*70)
        print("💡 EXAMPLE RECOMMENDATIONS")
        print("="*70)
        
        self.recommend_neighborhoods(budget=1500, n_rooms=2, n_bathrooms=1, top_n=5)
        
        print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")


# ============================================================================
# MAIN: Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Initialize
    predictor = RentalPredictor('data/houses_madrid.csv')
    
    # OPTION 1: Complete Pipeline
    print("\n📌 EXECUTING: COMPLETE PIPELINE")
    predictor.full_pipeline()
    
    # OPTION 2: Single Property Prediction
    print("\n\n📌 EXAMPLE: SINGLE PROPERTY PREDICTION")
    result = predictor.predict_single(
        sq_mt_built=85,
        sq_mt_useful=70,
        n_rooms=2,
        n_bathrooms=1,
        district='Salamanca'
    )
    if result:
        print(f"\n✅ Result: {result['label']}")
        print(f"   Confidence: {result['confidence_pct']}")
    
    # OPTION 3: District Analysis
    print("\n\n📌 EXAMPLE: DISTRICT ANALYSIS")
    predictor.get_district_analysis('Centro')
    
    # OPTION 4: District Comparison
    print("\n\n📌 EXAMPLE: TOP 5 DISTRICTS")
    predictor.compare_districts(
        districts=['Centro', 'Salamanca', 'Chamberí', 'Arganzuela', 'Retiro']
    )
    
    # OPTION 5: Recommendations
    print("\n\n📌 EXAMPLE: RECOMMENDATIONS BY BUDGET")
    predictor.recommend_neighborhoods(
        budget=1200,
        n_rooms=2,
        n_bathrooms=1,
        top_n=5
    )