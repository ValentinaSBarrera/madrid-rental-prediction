import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
import os

warnings.filterwarnings('ignore')


class RentalPriceModel:
    """Machine learning model to predict if a rental is expensive or fair"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.le_district = LabelEncoder()
        self.feature_columns = None
        self.district_stats = None
    
    def prepare_data(self, df_clean):
        """Prepare data for training"""
        print("📦 Preparing data...")
        
        X = df_clean[['sq_mt_built', 'sq_mt_useful', 'n_rooms', 'n_bathrooms']].copy()
        y = df_clean['overpriced'].copy()
        
        # Add district as feature
        X['district_encoded'] = self.le_district.fit_transform(df_clean['district'])
        
        # Fill NaN values
        X = X.fillna(X.mean())
        
        self.feature_columns = X.columns.tolist()
        
        # Save statistics by district
        self.district_stats = df_clean.groupby('district').agg({
            'rent_price': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'sq_mt_built': 'mean'
        }).round(2)
        
        print(f"   Features: {self.feature_columns}")
        print(f"   Shape: {X.shape}")
        print(f"   Balance: {y.value_counts().to_dict()}")
        print(f"   Districts: {len(self.district_stats)}")
        
        return X, y
    
    def train(self, df_clean):
        """Train classification model"""
        print("\n🎓 Training model...")
        
        X, y = self.prepare_data(df_clean)
        
        # 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train RandomForest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        print("\n✅ MODEL RESULTS:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Fair Price', 'Expensive']))
        
        # Feature importance
        print("\n🎯 Feature Importance:")
        for feat, importance in zip(self.feature_columns, self.model.feature_importances_):
            print(f"   {feat}: {importance:.3f}")
        
        return X_test, y_test, y_pred
    
    def predict(self, sq_mt_built, sq_mt_useful, n_rooms, n_bathrooms, district):
        """Predict if a rental is expensive or fair"""
        if self.model is None:
            raise ValueError("Model not trained. Run train() first.")
        
        try:
            district_encoded = self.le_district.transform([district])[0]
        except ValueError:
            return None, f"District '{district}' not found in training data"
        
        X_new = np.array([[sq_mt_built, sq_mt_useful, n_rooms, n_bathrooms, district_encoded]])
        X_new_scaled = self.scaler.transform(X_new)
        
        prediction = self.model.predict(X_new_scaled)[0]
        confidence = self.model.predict_proba(X_new_scaled)[0]
        
        # Get district statistics
        district_info = self._get_district_stats(district)
        
        result = {
            'overpriced': int(prediction),
            'label': 'EXPENSIVE 🔴' if prediction == 1 else 'FAIR PRICE 🟢',
            'confidence': float(max(confidence)),
            'confidence_pct': f"{max(confidence)*100:.1f}%",
            'avg_price': district_info['mean'],
            'median_price': district_info['median'],
            'min_price': district_info['min'],
            'max_price': district_info['max'],
            'std_price': district_info['std'],
            'num_properties': int(district_info['count']),
            # NUEVO: Precio por m²
            'price_per_sqm': round((district_info['mean'] / district_info['avg_sqm_built']), 2),
            'price_per_sqm_property': round(district_info['mean'] / sq_mt_built, 2),  # Del inmueble predeterminado
            'avg_sqm_district': district_info['avg_sqm_built'],
        }
        
        return result, None
    
    def _get_district_stats(self, district):
        """Get district statistics"""
        if self.district_stats is None:
            return {
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'std': 0,
                'count': 0,
                'avg_sqm_built': 0
            }
        
        try:
            stats = self.district_stats.loc[district]
            return {
                'mean': float(stats[('rent_price', 'mean')]),
                'median': float(stats[('rent_price', 'median')]),
                'min': float(stats[('rent_price', 'min')]),
                'max': float(stats[('rent_price', 'max')]),
                'std': float(stats[('rent_price', 'std')]),
                'count': float(stats[('rent_price', 'count')]),
                'avg_sqm_built': float(stats[('sq_mt_built', 'mean')])  # NUEVO
            }
        except:
            return {
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'std': 0,
                'count': 0,
                'avg_sqm_built': 0
            }
    
    def save(self, path='model/rental_model.pkl'):
        """Save trained model"""
        os.makedirs('model', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'le_district': self.le_district,
            'feature_columns': self.feature_columns,
            'district_stats': self.district_stats
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {path}")
    
    def load(self, path='model/rental_model.pkl'):
        """Load trained model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found. Run first: python src/model.py")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.le_district = model_data['le_district']
        self.feature_columns = model_data['feature_columns']
        self.district_stats = model_data.get('district_stats', None)
        print(f"✅ Model loaded from {path}")


# ============================================================================
# MAIN: Model Training
# ============================================================================

if __name__ == "__main__":
    """
    Train the Machine Learning model
    
    Requires:
        - File: data/madrid_rent_clean.csv (generated by etl.py)
    
    Generates:
        - File: model/rental_model.pkl
    """
    
    print("\n🚀 Starting ML Model Training")
    print("=" * 70)
    print("Dataset: madrid_rent_clean.csv")
    print("Model: Random Forest Classifier")
    print("=" * 70)
    
    try:
        # Load cleaned data
        print("\n📥 Loading cleaned dataset...")
        df_clean = pd.read_csv('data/madrid_rent_clean.csv')
        print(f"   Records: {len(df_clean):,}")
        print(f"   Columns: {len(df_clean.columns)}")
        
        # Train model
        print("\n" + "="*70)
        model = RentalPriceModel()
        model.train(df_clean)
        model.save('model/rental_model.pkl')
        
        print("\n" + "="*70)
        print("🎉 Model trained and saved successfully!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("   Make sure to run first: python src/etl.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()