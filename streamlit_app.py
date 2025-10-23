import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. PAGE CONFIG ---
st.set_page_config(
    layout="wide",
    page_title="Insurance Cost Predictor",
    page_icon="🩺"
)

# --- 2. ALL DATA LOADING & MODEL TRAINING (Happens once) ---

# We do all the "work" here at the top.
try:
    # --- A: Load Data ---
    df_raw = pd.read_csv("insurance.csv")
    
    # --- B: Process Data ---
    df_processed = df_raw.copy()
    df_processed['smoker'] = df_processed['smoker'].map({'yes': 1, 'no': 0})
    df_processed['sex'] = df_processed['sex'].map({'male': 1, 'female': 0})
    df_processed = pd.get_dummies(df_processed, columns=['region'], drop_first=True)

    # --- C: Define Features (X) and Target (y) ---
    features = ['age', 'bmi', 'children', 'smoker', 'sex', 'region_northwest', 'region_southeast', 'region_southwest']
    X = df_processed[features]
    y = df_processed['charges']

    # --- D: Split Data (80% train, 20% test) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- E: Train Model ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- F: Evaluate Model ---
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # --- G: Create DataFrames for Charts ---
    # For the Actual vs. Predicted chart
    eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    # For the coefficients table
    coefs = pd.DataFrame(model.coef_, index=features, columns=["Impact on Cost ($)"])

except FileNotFoundError:
    st.error("FATAL ERROR: 'insurance.csv' file not found. Make sure it's in a 'data' folder.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during data processing: {e}")
    st.stop()


# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Go to", 
    ["🏠 Business Case & Data", 
     "📊 Data Visualizations", 
     "🤖 Model Prediction & Evaluation"]
)


# --- 4. PAGE 1: BUSINESS CASE & DATA ---

if page == "🏠 Business Case & Data":
    st.title("Predicting Medical Costs 🩺")
    
    # --- FIX YOUR IMAGE PATH HERE ---
    st.image("img3.jpg", width=600) # <-- Make sure 'img3.jpg' is the correct name/path
    
    st.markdown("---")
    st.header("The Business Problem")
    st.write("""
    A health insurance company needs to set its yearly premiums. 
    To do this, they must accurately predict a client's potential medical costs. 
    If the price is too low, the company loses money. If it's too high, they lose customers.
    
    **Our goal:** Build a tool that predicts medical charges based on a client's profile.
    """)

    with st.expander("Click to see the Raw Data & Preparation"):
        st.subheader("Raw Data")
        st.write("This is the raw data we started with (1,338 rows).")
        st.dataframe(df_raw.head())
        st.markdown("Source: [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)")
        
        st.subheader("Data Preparation")
        st.write("""
        The model can't read text like 'male' or 'smoker'. We had to convert them to numbers:
        1. `sex` ('male'/'female') was changed to 1/0.
        2. `smoker` ('yes'/'no') was changed to 1/0.
        3. `region` (which has 4 categories) was converted into **3** columns. 
        This is a standard step called 'One-Hot Encoding' to avoid the 'dummy variable trap'. 
        The 4th region (`northeast`) is represented when the other 3 columns are all 0.
        """)
        st.write("This is the final processed data the model was trained on:")
        st.dataframe(df_processed.head())


# --- 5. PAGE 2: DATA VISUALIZATIONS ---

elif page == "📊 Data Visualizations":
    st.title("Visual Insights: What Drives Medical Costs? 📈")
    st.markdown("---")

    # Use tabs to look professional, like your professor's example
    tab1, tab2 = st.tabs(["Insight 1: Smoking", "Insight 2: BMI"])

    with tab1:
        st.header("Smoking is the single biggest factor.")
        st.write("Non-smokers (blue) have low costs. Smokers (orange) are in a completely different, high-cost category.")
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        # Use the raw data for this chart so 'smoker' is 'yes'/'no'
        sns.scatterplot(data=df_raw, x='age', y='charges', hue='smoker', alpha=0.7, ax=ax1)
        ax1.set_title("Age vs. Charges by Smoking Status")
        st.pyplot(fig1)
    
    with tab2:
        st.header("BMI (Body Mass Index) also has a clear impact.")
        st.write("Higher BMI is generally associated with higher medical costs, especially for smokers.")
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_raw, x='bmi', y='charges', hue='smoker', alpha=0.7, ax=ax2)
        ax2.set_title("BMI vs. Charges by Smoking Status")
        st.pyplot(fig2)

# --- 6. PAGE 3: MODEL PREDICTION & EVALUATION ---

elif page == "🤖 Model Prediction & Evaluation":
    
    # --- Prediction Tool ---
    st.title("Insurance Premium Calculator 🧮")
    st.write("Use this tool to predict costs for a new client.")
    
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
        # Convert user's text inputs to numbers
        smoker_val = 1 if smoker_text == "Yes" else 0
        sex_val = 1 if sex_text == "Male" else 0
        
        # Handle One-Hot Encoding for region
        region_northwest = 1 if region_text == "northwest" else 0
        region_southeast = 1 if region_text == "southeast" else 0
        region_southwest = 1 if region_text == "southwest" else 0
        
        # Create the input DataFrame
        input_data = pd.DataFrame([[age, bmi, children, smoker_val, sex_val, 
                                    region_northwest, region_southeast, region_southwest]], 
                                  columns=features)
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Display prediction
        st.metric(label="Predicted Yearly Cost", value=f"${prediction[0]:,.2f}")

    st.markdown("---")

    # --- Model Evaluation (like the Superstore example) ---
    st.title("Model Evaluation 🔬")
    st.write("This shows how well our model performed on the 20% 'test set' (data it had never seen).")

    # Show metrics in columns
    col_r2, col_mae = st.columns(2)
    with col_r2:
        st.metric(label="R-squared (Model Fit)", value=f"{r2:.2f}")
    with col_mae:
        st.metric(label="Mean Absolute Error (MAE)", value=f"${mae:,.2f}")

    st.info(f"""
    **Interpretation:**
    * **R-squared:** Our model explains **{r2:.0%}** of the variation in medical costs.
    * **MAE:** On average, our model's prediction is off by **${mae:,.2f}**.
    """)

    st.header("Actual vs. Predicted Costs")
    st.write("This chart plots the *Actual* cost (x-axis) against the cost our model *Predicted* (y-axis). A perfect model would have all dots on the red line.")
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=eval_df, x='Actual', y='Predicted', alpha=0.7, ax=ax3)
    # Add the 45-degree "perfect prediction" line
    min_val = min(eval_df['Actual'].min(), eval_df['Predicted'].min())
    max_val = max(eval_df['Actual'].max(), eval_df['Predicted'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    st.pyplot(fig3)

    # --- Model Coefficients ---
    st.header("What Did Our Model Learn?")
    st.write("These 'coefficients' tell us how much each factor impacts the final cost.")
    st.dataframe(coefs.round(2))
    
    st.success(f"""
    **Key Takeaway:**
    Being a smoker is the biggest factor, adding an estimated **${coefs.loc['smoker'][0]:,.2f}** to your yearly costs!
    """)