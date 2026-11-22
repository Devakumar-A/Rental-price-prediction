import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏠 AI-Powered Housing Price Predictor")
st.markdown(
    "Predict house and rental prices in real-time and visualize data correlations interactively."
)

# -------------------------------
# Load Trained Model
# -------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Load dataset (for visualization)
@st.cache_data
def load_data():
    return pd.read_csv("data/Housing.csv")

df = load_data()

# -------------------------------
# User Input Section
# -------------------------------
st.sidebar.header("Property Details")

# Region / City selection
city = st.sidebar.selectbox(
    "Preferred City / Region",
    [
        'Pondicherry', 'Chennai', 'Coimbatore', 'Madurai', 'Salem', 'Tiruchirappalli',
        'Erode', 'Tiruppur', 'Vellore', 'Thanjavur', 'Dindigul', 'Nagercoil',
        'Karur', 'Kanchipuram', 'Viluppuram', 'Cuddalore', 'Nagapattinam',
        'Ramanathapuram', 'Tirunelveli', 'Sivaganga', 'Theni', 'Krishnagiri',
        'Perambalur', 'Ariyalur', 'Tiruvarur', 'Namakkal', 'Dharmapuri',
        'Nilgiris', 'Kanyakumari'
    ]
)

area = st.sidebar.number_input("Area (sq ft)", min_value=100, max_value=50000, value=7500, step=50)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=5, value=2, step=1)
stories = st.sidebar.number_input("Stories (Floors)", min_value=1, max_value=5, value=2, step=1)
parking = st.sidebar.number_input("Parking Slots", min_value=0, max_value=10, value=1, step=1)

mainroad = st.sidebar.selectbox("Main Road Access", ['yes', 'no'])
guestroom = st.sidebar.selectbox("Guest Room", ['yes', 'no'])
basement = st.sidebar.selectbox("Basement", ['yes', 'no'])
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ['yes', 'no'])
airconditioning = st.sidebar.selectbox("Air Conditioning", ['yes', 'no'])
prefarea = st.sidebar.selectbox("Preferred / Prime Locality", ['yes', 'no'])
furnishingstatus = st.sidebar.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

# Define the exact order of columns from the trained model
model_columns = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'mainroad_yes', 'guestroom_yes', 'basement_yes',
    'hotwaterheating_yes', 'airconditioning_yes', 'prefarea_yes',
    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
]

# Map inputs to model features
input_data = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'mainroad_yes': 1 if mainroad == 'yes' else 0,
    'guestroom_yes': 1 if guestroom == 'yes' else 0,
    'basement_yes': 1 if basement == 'yes' else 0,
    'hotwaterheating_yes': 1 if hotwaterheating == 'yes' else 0,
    'airconditioning_yes': 1 if airconditioning == 'yes' else 0,
    'parking': parking,
    'prefarea_yes': 1 if prefarea == 'yes' else 0,
    'furnishingstatus_semi-furnished': 1 if furnishingstatus == 'semi-furnished' else 0,
    'furnishingstatus_unfurnished': 1 if furnishingstatus == 'unfurnished' else 0
}

# Create DataFrame with exact column order
input_df = pd.DataFrame([input_data])[model_columns]

# -------------------------------
# Prediction
# -------------------------------
predicted_price = model.predict(input_df)[0]

st.markdown("## Predicted Price")
col1, col2 = st.columns([1, 2])
with col1:
    st.success(f"₹ {predicted_price:,.0f} / YEARLY")
    st.info(f"Approx. ₹ {predicted_price / 12:,.0f} / MONTHLY")
with col2:
    st.write(f"**Selected Region:** {city}")
    st.write(f"**Specifications:** {area:,} sq ft | {bedrooms} BHK | {bathrooms} Bath | {furnishingstatus.title()}")

# -------------------------------
# Visualizations
# -------------------------------
st.markdown("---")
st.markdown("## Interactive Visualizations")

tab1, tab2 = st.tabs(["Price Distribution", "Correlation Heatmap"])

with tab1:
    fig_price = px.histogram(
        df,
        x='price',
        nbins=30,
        opacity=0.75,
        color_discrete_sequence=['#1f77b4'],
        marginal="box",
        title="Housing Price Distribution with Predicted Value Marker"
    )
    fig_price.add_vline(
        x=predicted_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Predicted: ₹ {predicted_price:,.0f}",
        annotation_position="top right"
    )
    fig_price.update_layout(
        xaxis_title="Price (₹)",
        yaxis_title="Frequency",
        bargap=0.15,
        template="plotly_white"
    )
    st.plotly_chart(fig_price, use_container_width=True)

with tab2:
    corr = df.corr(numeric_only=True)
    fig_corr = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.columns),
        annotation_text=np.round(corr.values, 2),
        colorscale='RdBu',
        showscale=True,
        reversescale=True,
        zmin=-1,
        zmax=1
    )
    fig_corr.update_layout(
        autosize=True,
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text="Numerical Feature Correlation Heatmap", x=0.5, xanchor='center'),
        xaxis=dict(tickangle=45, automargin=True),
        yaxis=dict(automargin=True),
        template="plotly_white"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made by RGCET TECHNOCRATS</p>", unsafe_allow_html=True)
