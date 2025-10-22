import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set page config and header
st.set_page_config(
    layout="wide",
    page_title="Insurance Cost Predictor",
    page_icon="🩺"
)

# Function to load and cache the raw data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

# Function to load, process, and cache the data for the model
@st.cache_data
def load_and_prep_data(path):
    df = pd.read_csv(path)
    df_processed = df.copy()
    df_processed['smoker'] = df_processed['smoker'].map({'yes': 1, 'no': 0})
    df_processed['sex'] = df_processed['sex'].map({'male': 1, 'female': 0})
    df_processed = pd.get_dummies(df_processed, columns=['region'], drop_first=True)
    return df_processed

# Function to train the model and cache it
@st.cache_data
def train_model(df):
    features = ['age', 'bmi', 'children', 'smoker', 'sex', 'region_northwest', 'region_southeast', 'region_southwest']
    X = df[features]
    y = df['charges']
    model = LinearRegression()
    model.fit(X, y)
    return model, features