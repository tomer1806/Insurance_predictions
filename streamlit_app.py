import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- 1. PAGE CONFIG ---
# Setup the main page settings
st.set_page_config(
    layout="wide",
    page_title="Insurance Cost Predictor",
    page_icon="🩺"
)

# --- 2. ALL DATA LOADING & MODEL TRAINING ---
# This block loads, processes, and trains the model once
df_raw = pd.read_csv("insurance.csv") 

df_processed = df_raw.copy()
df_processed['smoker'] = df_processed['smoker'].map({'yes': 1, 'no': 0})
df_processed['sex'] = df_processed['sex'].map({'male': 1, 'female': 0})
df_processed = pd.get_dummies(df_processed, columns=['region'], drop_first=True)

# feature engineering
df_processed['bmi_smoker'] = df_processed['bmi'] * df_processed['smoker']
df_processed['age_smoker'] = df_processed['age'] * df_processed['smoker']

# define features (X) and target (y)
features = [
    'age', 'bmi', 'children', 'smoker', 'sex', 
    'age_smoker', 'bmi_smoker', 
    'region_northwest', 'region_southeast', 'region_southwest'
]
X = df_processed[features]
y = df_processed['charges']

# split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) 

# prep dataframes for charts
eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
coefs = pd.DataFrame(model.coef_, index=features, columns=["Impact on Cost ($)"])
train_df = X_train.copy()
train_df['charges'] = y_train


# --- 3. SIDEBAR NAVIGATION ---
# Create the sidebar menu
st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Go to", 
    ["🏠 Business Case & Data", 
     "📊 Data Visualizations", 
     "🤖 Model Prediction & Evaluation"]
)


# --- 4. PAGE 1: BUSINESS CASE & DATA ---
# Logic for showing the first page
if page == "🏠 Business Case & Data":
    st.title("Predicting Medical Costs 🩺")
    
    # Display the header image (assumes 'img3.jpg' is in the main folder)
    st.image("img3.jpg", width=600) 
    
    st.markdown("---") 
    st.header("The Business Problem")
    st.write("""
    A health insurance company needs to set its yearly premiums. 
    To do this, they must accurately predict a client's potential medical costs. 
    If the price is too low, the company loses money. If it's too high, they lose customers.
    
    **Our goal:** Build a tool that predicts medical charges based on a client's profile
    """)

    with st.expander("Click to see the Raw Data & Preparation"):
        st.subheader("Raw Data")
        st.write("This is the raw data we started with (1,338 rows)")
        st.dataframe(df_raw.head())
        st.markdown("Source: [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)")
        
        st.subheader("Data Preparation (Feature Engineering)")
 
        st.markdown("##### Step 1: Converting Text to Numbers")
        st.write("""
        A model can't understand text like 'male' or 'northeast', so our first step was to convert these categories into numbers.
        
        * `sex` ('female'/'male') was converted to 0 / 1.
        * `smoker` ('no'/'yes') was also converted to 0 / 1.
        * `region` (which had 4 areas) was converted into 3 new binary columns using One-Hot Encoding. (We drop one to avoid multicollinearity).
        """)
        
        st.markdown("##### Step 2: Creating Interaction Features")
        st.write("""
        To make the model smarter, we added "combo" features based on our insights from the charts:
        """)
        st.markdown("""
        1. **`bmi_smoker` (bmi * smoker):** This teaches the model that the cost of high BMI is *worse* for smokers.
        2. **`age_smoker` (age * smoker):** This teaches the model that the cost of smoking gets *worse* with age.
        """)
        
        st.write("This is the final processed data the model was trained on:")
        st.dataframe(df_processed.head())


# --- 5. PAGE 2: DATA VISUALIZATIONS ---
# Logic for showing the second page
elif page == "📊 Data Visualizations":
    st.title("Visual Insights: What Drives Medical Costs? 📈")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Insight 1: Smoking", 
        "Insight 2: BMI", 
        "Descriptive Statistics", 
        "Correlation Heatmap"
    ])

    with tab1:
        st.header("Smoking is the single biggest factor")
        st.write("Non-smokers (Orange) have low costs. Smokers (Blue) are in a completely different, high-cost category")
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_raw, x='age', y='charges', hue='smoker', alpha=0.7, ax=ax1)
        ax1.set_title("Age vs. Charges by Smoking Status")
        st.pyplot(fig1)
    
    with tab2:
        st.header("BMI (Body Mass Index) also has a clear impact on smokers")
        st.write("This chart shows that BMI has a complex effect. For non-smokers, costs only jump significantly once BMI passes a high threshold (around 30). For smokers, however, any increase in BMI is linked to a steep rise in cost")
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_raw, x='bmi', y='charges', hue='smoker', alpha=0.7, ax=ax2)
        ax2.set_title("BMI vs. Charges by Smoking Status")
        st.pyplot(fig2)

    with tab3:
        st.header("Descriptive Statistics")
        st.write("Here are the mean, min, max, and standard deviation for our raw data")
        st.dataframe(df_raw[['age', 'bmi', 'children', 'charges']].describe())

    with tab4:
        st.header("Feature Correlation Heatmap")
        st.write("This map shows how strongly each feature is related to 'charges'")
        
        corr = df_processed.corr()
        
        fig3, ax3 = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)


# --- 6. PAGE 3: MODEL PREDICTION & EVALUATION ---
# Logic for showing the third page
elif page == "🤖 Model Prediction & Evaluation":
    
    st.title("Insurance Premium Calculator 🧮")
    st.write("Use this tool to predict costs for a new client")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Select Age", 18, 100, 30)
        bmi = st.slider("Select BMI", 15.0, 55.0, 25.0, 0.1)
        children = st.slider("Select Number of Children", 0, 5, 0)
    with col2:
        smoker_text = st.selectbox("Is the client a smoker?", ["No", "Yes"])
        sex_text = st.selectbox("What is the client's sex?", ["Female", "Male"])
        region_text = st.selectbox("What is the client's region?", ["southwest", "southeast", "northwest", "northeast"])
    
    if st.button("Predict Costs", type="primary"):
        # convert user inputs to numbers
        smoker_val = 1 if smoker_text == "Yes" else 0
        sex_val = 1 if sex_text == "Male" else 0
        
        # handle region encoding
        region_northwest = 1 if region_text == "northwest" else 0
        region_southeast = 1 if region_text == "southeast" else 0
        region_southwest = 1 if region_text == "southwest" else 0
        
        # create the interaction features
        age_smoker_val = age * smoker_val
        bmi_smoker_val = bmi * smoker_val
        
        # build the input row for prediction
        input_data = pd.DataFrame([[
            age, bmi, children, smoker_val, sex_val, 
            age_smoker_val, bmi_smoker_val, 
            region_northwest, region_southeast, region_southwest
        ]], columns=features)
        
        prediction = model.predict(input_data)
        
        st.subheader("Prediction Results")
        st.metric(label="Model's Predicted Cost", value=f"${prediction[0]:,.2f}")

    st.markdown("---")

    st.title("Model Evaluation 🔬")
    st.write("This shows how well our model performed on the 20% 'test set'")

    col_r2, col_mae, col_rmse = st.columns(3)
    with col_r2:
        st.metric(label="R-squared (Model Fit)", value=f"{r2:.3f}") 
    with col_mae:
        st.metric(label="Mean Absolute Error (MAE)", value=f"${mae:,.2f}") 
    with col_rmse:
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"${rmse:,.2f}") 

    st.info(f"""
    **Interpretation:**
    * **R-squared (R²):** Our model explains **{r2:.1%}** of the variation in medical costs.
    * **Mean Absolute Error (MAE):** On average, our model's prediction is off by **${mae:,.2f}**.
    """)

    st.header("Actual vs. Predicted Costs (Test Data)")
    st.write("The red line is a 'perfect' prediction. Our model's predictions (blue dots) follow this line very closely")
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=eval_df, x='Actual', y='Predicted', alpha=0.7, ax=ax3)
    min_val = min(eval_df['Actual'].min(), eval_df['Predicted'].min())
    max_val = max(eval_df['Actual'].max(), eval_df['Predicted'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2) 
    st.pyplot(fig3)

    st.header("What Did Our Model Learn?")
    st.write("These 'coefficients' tell us how much each factor impacts the final cost")
    st.dataframe(coefs.round(2))
    
