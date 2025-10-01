# import streamlit as st
# import pickle
# import pandas as pd

# # Load model
# with open("banglore_home_prices_model.pickle", "rb") as f:
#     model = pickle.load(f)


# # Load columns (your saved JSON)
# import json
# with open("columns.json", "r") as f:
#     data_columns = json.load(f)["data_columns"]

# st.title("🏠 Bangalore House Price Prediction App")

# # Inputs
# sqft = st.number_input("Total Square Feet", min_value=500, max_value=5000, value=1000)
# bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2)
# bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
# location = st.selectbox("Location", data_columns[3:])  # first 3 are sqft, bath, bhk

# if st.button("Predict Price"):
#     # Convert input into model format
#     x = pd.DataFrame([[sqft, bath, bhk] + [0]*(len(data_columns)-3)], columns=data_columns)
#     if location in data_columns:
#         loc_index = data_columns.index(location)
#         x.iloc[0, loc_index] = 1
    
#     prediction = model.predict(x)[0]
#     st.success(f"Estimated Price: ₹ {prediction:.2f} lakhs")



import streamlit as st
import pickle
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# ----------------------------
# Load Models
# ----------------------------
models = {}
for name in ["linear_regression", "lasso", "decision_tree"]:
    with open(f"{name}_best_model.pickle", "rb") as f:
        models[name] = pickle.load(f)

# Load columns
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# Load original dataset (with location) for EDA
df_original = pd.read_csv("bangalore_home_prices_original.csv")

# Load processed dataset (after one-hot encoding) for model training/prediction
df = pd.read_csv("bangalore_home_prices.csv")

# ----------------------------
# Streamlit App
# ----------------------------
st.title("🏠 Bangalore House Price Prediction & Analysis")

tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Data Analysis", "🤖 Model Comparison"])

# ----------------------------
# Tab 1: Prediction
# ----------------------------
with tab1:
    st.header("Predict House Price")
    sqft = st.number_input("Total Square Feet", min_value=500, max_value=5000, value=1000)
    bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2)
    bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    location = st.selectbox("Location", data_columns[3:])  # skip first 3
    
    model_name = st.selectbox("Select Model", ["Linear Regression", "Lasso", "Decision Tree"])

    if st.button("Predict Price"):
        x = pd.DataFrame([[sqft, bath, bhk] + [0]*(len(data_columns)-3)], columns=data_columns)
        if location in data_columns:
            loc_index = data_columns.index(location)
            x.iloc[0, loc_index] = 1

        # Map selectbox value to your dictionary keys
        model_key = model_name.lower().replace(" ", "_")
        prediction = models[model_key].predict(x)[0]
        st.success(f"Estimated Price: ₹ {prediction:.2f} lakhs")

# ----------------------------
# Tab 2: Data Analysis
# ----------------------------
with tab2:
    st.header("Exploratory Data Analysis")
    
    # Distribution of Prices
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_original['price'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)
    
    # Scatter: Sqft vs Price
    st.subheader("Price vs Square Feet")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df_original['total_sqft'], y=df_original['price'], 
                    hue=df_original['bhk'], palette="viridis", ax=ax)
    st.pyplot(fig)
    
    # Average Price by Location (Top 10)
    st.subheader("Top 10 Locations by Avg Price")
    loc_price = df_original.groupby("location")["price"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    loc_price.plot(kind="bar", ax=ax)
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df_original[["total_sqft", "bath", "bhk", "price"]].corr(),
                annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ----------------------------
# Tab 3: Model Comparison
# ----------------------------
with tab3:
    st.header("Compare Models")

    # Train/test split for evaluation
    from sklearn.model_selection import train_test_split

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results.append({
            "Model": name.title(),
            "R² Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Bar chart of R²
    st.subheader("Model R² Comparison")
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="R² Score", data=results_df, ax=ax)
    st.pyplot(fig)
