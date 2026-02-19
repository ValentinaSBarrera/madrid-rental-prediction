# International translations for Madrid Rental Price Predictor
TRANSLATIONS = {
    'es': {
        # Header & Navigation
        'language': 'Idioma',
        'spanish': 'Español',
        'english': 'English',
        
        # Main Title & Subtitle
        'title': '🏠 Predictor de Precios de Alquiler en Madrid',
        'subtitle': '¿Es un buen precio? Descúbrelo en segundos',
        
        # Objective Section
        'objective_title': '🎯 Objetivo',
        'objective_text': 'Predecir si un inmueble en alquiler en Madrid está caro o a buen precio usando Machine Learning y datos reales de Kaggle.',
        
        # Error Messages
        'error_model_not_found': 'Modelo no encontrado. Ejecuta primero:',
        'error_loading_model': 'Error al cargar el modelo',
        
        # About Section
        'about_project': 'Sobre este proyecto',
        'dataset_source': 'Dataset',
        'dataset_value': 'Kaggle - Madrid Housing Prices',
        'model_type': 'Modelo',
        'model_value': 'Random Forest Classifier',
        'prediction_objective': 'Objetivo',
        'prediction_objective_value': 'Predecir si un alquiler está caro o barato',
        
        # Statistics Section
        'statistics': 'Estadísticas',
        'districts': 'Distritos',
        'neighborhoods': 'Barrios',
        'records': 'Registros',
        'districts_count': '21',
        'neighborhoods_count': '+130',
        'records_count': '+4,000',
        
        # Input Section
        'input_title': '📋 Características del Inmueble',
        'built_sqm': '📐 Metros cuadrados construidos',
        'bedrooms': '🛏️ Número de habitaciones',
        'useful_sqm': '📏 Metros cuadrados útiles',
        'bathrooms': '🚿 Número de baños',
        'district': '📍 Distrito',
        
        # Button
        'predict_btn': '🔮 Predecir precio',
        
        # Results Section
        'prediction': 'Predicción',
        'confidence': 'Confianza',
        'avg_price': '💰 Precio Medio',
        'price_per_sqm': '📊 Precio por m²',
        'properties': '📊 Propiedades',
        'details': 'Detalles de la predicción',
        
        # Property Details
        'property_features': '🏠 Características del inmueble',
        'district_stats': '📈 Estadísticas del distrito',
        'price_per_sqm_property': 'Precio/m² de este inmueble',
        'avg_price_per_sqm': 'Precio promedio/m² del distrito',
        
        # Price Labels
        'fair_price': 'PRECIO JUSTO 🟢',
        'expensive': 'CARO 🔴',
        
        # Interpretations
        'fair_interpretation': 'Este piso tiene un **PRECIO JUSTO**. El precio está **por debajo o cerca** del promedio del distrito. Es una buena oportunidad.',
        'expensive_interpretation': 'Este piso probablemente está **CARO**. El precio está **por encima** del promedio del distrito. Considera negociar o buscar alternativas.',
        
        # Statistics Labels
        'avg': 'Promedio',
        'median': 'Mediana',
        'min': 'Mínimo',
        'max': 'Máximo',
        'avg_sqm_district': 'Promedio m² del distrito',
        'month': '/mes',
        'sqm': 'm²',
        'per_sqm': '€/m²',
        
        # Footer
        'portfolio': 'Proyecto de Portfolio',
        'github': 'GitHub',
        'data_science': 'Data Science 2026',
    },
    'en': {
        # Header & Navigation
        'language': 'Language',
        'spanish': 'Español',
        'english': 'English',
        
        # Main Title & Subtitle
        'title': '🏠 Madrid Rental Price Predictor',
        'subtitle': 'Is it a fair price? Find out in seconds',
        
        # Objective Section
        'objective_title': '🎯 Objective',
        'objective_text': 'Predict whether a rental property in Madrid is expensive or fairly priced using Machine Learning and real Kaggle data.',
        
        # Error Messages
        'error_model_not_found': 'Model not found. Run first:',
        'error_loading_model': 'Error loading model',
        
        # About Section
        'about_project': 'About this project',
        'dataset_source': 'Dataset',
        'dataset_value': 'Kaggle - Madrid Housing Prices',
        'model_type': 'Model',
        'model_value': 'Random Forest Classifier',
        'prediction_objective': 'Objective',
        'prediction_objective_value': 'Predict if a rental is expensive or fair',
        
        # Statistics Section
        'statistics': 'Statistics',
        'districts': 'Districts',
        'neighborhoods': 'Neighborhoods',
        'records': 'Records',
        'districts_count': '21',
        'neighborhoods_count': '+130',
        'records_count': '+4,000',
        
        # Input Section
        'input_title': '📋 Property Features',
        'built_sqm': '📐 Built square meters',
        'bedrooms': '🛏️ Number of bedrooms',
        'useful_sqm': '📏 Useful square meters',
        'bathrooms': '🚿 Number of bathrooms',
        'district': '📍 District',
        
        # Button
        'predict_btn': '🔮 Predict price',
        
        # Results Section
        'prediction': 'Prediction',
        'confidence': 'Confidence',
        'avg_price': '💰 Average Price',
        'price_per_sqm': '📊 Price per m²',
        'properties': '📊 Properties',
        'details': 'Prediction Details',
        
        # Property Details
        'property_features': '🏠 Property Features',
        'district_stats': '📈 District Statistics',
        'price_per_sqm_property': 'Price/m² of this property',
        'avg_price_per_sqm': 'Average price/m² in district',
        
        # Price Labels
        'fair_price': 'FAIR PRICE 🟢',
        'expensive': 'EXPENSIVE 🔴',
        
        # Interpretations
        'fair_interpretation': 'This property has a **FAIR PRICE**. The price is **below or near** the district average. It\'s a good opportunity.',
        'expensive_interpretation': 'This property is probably **EXPENSIVE**. The price is **above** the district average. Consider negotiating or searching for alternatives.',
        
        # Statistics Labels
        'avg': 'Average',
        'median': 'Median',
        'min': 'Min',
        'max': 'Max',
        'avg_sqm_district': 'Average m² in district',
        'month': '/month',
        'sqm': 'm²',
        'per_sqm': '€/m²',
        
        # Footer
        'portfolio': 'Portfolio Project',
        'github': 'GitHub',
        'data_science': 'Data Science 2026',
    }
}

def get_text(key, language='en'):
    """Get translated text by key and language"""
    return TRANSLATIONS.get(language, TRANSLATIONS['en']).get(key, key)