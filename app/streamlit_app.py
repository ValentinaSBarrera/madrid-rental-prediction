import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src/ folder to PATH
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import RentalPriceModel
from i18n import get_text

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Madrid Rental Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Language Selection
# ============================================================================

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🇪🇸 Español", use_container_width=True, key="es_btn"):
        st.session_state.language = 'es'
with col2:
    if st.button("🇬🇧 English", use_container_width=True, key="en_btn"):
        st.session_state.language = 'en'

# Initialize language
if 'language' not in st.session_state:
    st.session_state.language = 'en'

lang = st.session_state.language

# ============================================================================
# Helper Function
# ============================================================================

def t(key):
    """Shortcut to get translated text"""
    return get_text(key, lang)

# ============================================================================
# Title 
# ============================================================================

st.title(t('title'))
st.markdown(t('subtitle'))


# ============================================================================
# Load Model
# ============================================================================

@st.cache_resource
def load_model():
    model = RentalPriceModel()
    model.load('model/rental_model.pkl')
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"❌ {t('error_model_not_found')}")
    st.code("python src/etl.py\npython src/model.py")
    st.stop()
except Exception as e:
    st.error(f"❌ {t('error_loading_model')}: {str(e)}")
    st.stop()

# ============================================================================
# Sidebar Information
# ============================================================================

st.sidebar.markdown(f"""
### {t('about_project')}
- **{t('dataset_source')}:** {t('dataset_value')}
- **{t('model_type')}:** {t('model_value')}
- **{t('prediction_objective')}:** {t('objective_text')}

### {t('statistics')}
- **{t('districts')}:** {t('districts_count')}
- **{t('neighborhoods')}:** {t('neighborhoods_count')}
- **{t('records')}:** {t('records_count')}
""")

# ============================================================================
# Input Section
# ============================================================================

st.markdown("---")
st.subheader(t('input_title'))

col1, col2 = st.columns(2)

with col1:
    sq_mt_built = st.number_input(
        t('built_sqm'),
        min_value=20, max_value=300, value=80, step=5
    )
    n_rooms = st.selectbox(
        t('bedrooms'),
        options=[1, 2, 3, 4, 5, 6]
    )

with col2:
    sq_mt_useful = st.number_input(
        t('useful_sqm'),
        min_value=15, max_value=250, value=70, step=5
    )
    n_bathrooms = st.selectbox(
        t('bathrooms'),
        options=[1, 2, 3, 4]
    )

# Districts list
districts = [
    'Centro', 'Arganzuela', 'Retiro', 'Salamanca', 'Chamberí',
    'Tetuán', 'Chamartín', 'Fuencarral-El Pardo', 'Moncloa-Aravaca',
    'Latina', 'Carabanchel', 'Usera', 'Puente de Vallecas', 'Moratalaz',
    'Ciudad Lineal', 'Hortaleza', 'Villaverde', 'Villa de Vallecas',
    'Vicálvaro', 'San Blas-Canillejas', 'Barajas'
]

district = st.selectbox(
    t('district'),
    options=districts
)

# ============================================================================
# Prediction
# ============================================================================

st.markdown("---")

if st.button(t('predict_btn'), use_container_width=True, key="predict"):
    result, error = model.predict(sq_mt_built, sq_mt_useful, n_rooms, n_bathrooms, district)
    
    if error:
        st.error(f"⚠️ {error}")
    else:
        # CORREGIDO: Traducir el label según el idioma actual
        if result['overpriced'] == 1:
            prediction_label = t('expensive')  # CARO 🔴 o EXPENSIVE 🔴
        else:
            prediction_label = t('fair_price')  # PRECIO JUSTO 🟢 o FAIR PRICE 🟢
        
        # Métricas (4 columnas)
        col1, col2, col3, col4 = st.columns(4)
        
        # Métrica 1: Predicción (CON TRADUCCIÓN CORRECTA)
        with col1:
            st.metric(
                label=t('prediction'),
                value=prediction_label,  # AHORA USA LA TRADUCCIÓN CORRECTA
                delta=f"{t('confidence')}: {result['confidence_pct']}"
            )
        
        # Métrica 2: Precio Promedio
        with col2:
            st.metric(
                label=t('avg_price'),
                value=f"€{result['avg_price']:.0f}",
                delta=f"{t('median')}: €{result['median_price']:.0f}"
            )
        
        # Métrica 3: Precio por m²
        with col3:
            st.metric(
                label=t('price_per_sqm'),
                value=f"€{result['price_per_sqm']:.2f}",
                delta=f"{t('avg_sqm_district')}: {result['avg_sqm_district']:.0f}{t('sqm')}"
            )
        
        # Métrica 4: Número de Propiedades
        with col4:
            st.metric(
                label=t('properties'),
                value=f"{result['num_properties']:,}",
                delta=f"σ: €{result['std_price']:.0f}"
            )
        
        # Detailed Information
        st.markdown("---")
        st.subheader(t('details'))
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown(f"#### {t('property_features')}")
            
            # Extract labels from translated text
            built_sqm_label = t('built_sqm').split('📐 ')[1] if '📐' in t('built_sqm') else t('built_sqm')
            useful_sqm_label = t('useful_sqm').split('📏 ')[1] if '📏' in t('useful_sqm') else t('useful_sqm')
            bedrooms_label = t('bedrooms').split('🛏️ ')[1] if '🛏️' in t('bedrooms') else t('bedrooms')
            bathrooms_label = t('bathrooms').split('🚿 ')[1] if '🚿' in t('bathrooms') else t('bathrooms')
            district_label = t('district').split('📍 ')[1] if '📍' in t('district') else t('district')
            
            details1 = f"""
- **{built_sqm_label}:** {sq_mt_built} {t('sqm')}
- **{useful_sqm_label}:** {sq_mt_useful} {t('sqm')}
- **{bedrooms_label}:** {n_rooms}
- **{bathrooms_label}:** {n_bathrooms}
- **{district_label}:** {district}
- **{t('price_per_sqm_property')}:** €{result['price_per_sqm_property']:.2f}{t('per_sqm')}
            """
            st.info(details1)
        
        with col_right:
            st.markdown(f"#### {t('district_stats')}")
            details2 = f"""
- **{t('avg')}:** €{result['avg_price']:.2f}{t('month')}
- **{t('median')}:** €{result['median_price']:.2f}{t('month')}
- **{t('min')}:** €{result['min_price']:.2f}{t('month')}
- **{t('max')}:** €{result['max_price']:.2f}{t('month')}
- **{t('avg_price_per_sqm')}:** €{result['price_per_sqm']:.2f}{t('per_sqm')}
- **{t('properties')}:** {result['num_properties']:,}
- **{t('confidence')}:** {result['confidence_pct']}
            """
            st.success(details2)
        
        # Interpretation - AHORA CON TRADUCCIÓN CORRECTA
        st.markdown("---")
        if result['overpriced'] == 1:
            st.warning(f"""
### {t('expensive')}

{t('expensive_interpretation')}
            """)
        else:
            st.success(f"""
### {t('fair_price')}

{t('fair_interpretation')}
            """)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center'>
    <small>🚀 {t('portfolio')} | {t('github')}: ValentinaSBarrera | {t('data_science')}</small>
</div>
""", unsafe_allow_html=True)