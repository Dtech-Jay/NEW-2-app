import streamlit as st
import joblib
import pandas as pd

# --------------------------------------------------
# 1. Load trained model and preprocessing objects
# --------------------------------------------------
model = joblib.load("linear_regression_model.joblib")
scaler = joblib.load("standard_scaler.joblib")
encoder = joblib.load("one_hot_encoder.joblib")

# --------------------------------------------------
# 2. App title
# --------------------------------------------------
st.title("Total Revenue Prediction App")
st.write("Enter product and customer details to predict total revenue.")

# --------------------------------------------------
# 3. User inputs
# --------------------------------------------------
st.header("Product Details")

price = st.number_input("Price", min_value=0.0, value=100.0)
discount_percent = st.number_input("Discount Percent", min_value=0, max_value=100, value=10)
quantity_sold = st.number_input("Quantity Sold", min_value=1, value=1)
rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.1)
review_count = st.number_input("Review Count", min_value=0, value=100)

product_category = st.selectbox(
    "Product Category",
    encoder.categories_[0]
)

st.header("Customer Details")

customer_region = st.selectbox(
    "Customer Region",
    encoder.categories_[1]
)

payment_method = st.selectbox(
    "Payment Method",
    encoder.categories_[2]
)

# --------------------------------------------------
# 4. Prediction
# --------------------------------------------------
if st.button("Predict Total Revenue"):

    # ---- Create raw input dataframe ----
    input_df = pd.DataFrame({
        "price": [price],
        "discount_percent": [discount_percent],
        "quantity_sold": [quantity_sold],
        "rating": [rating],
        "review_count": [review_count],
        "discounted_price": [price * (1 - discount_percent / 100)],
        "product_category": [product_category],
        "customer_region": [customer_region],
        "payment_method": [payment_method]
    })

    # ---- Encode categorical features ----
    categorical_cols = ["product_category", "customer_region", "payment_method"]
    encoded_array = encoder.transform(input_df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # ---- Combine numerical + encoded features ----
    numerical_cols = [
        "price",
        "discount_percent",
        "quantity_sold",
        "rating",
        "review_count",
        "discounted_price"
    ]

    processed_input = pd.concat(
        [input_df[numerical_cols].reset_index(drop=True), encoded_df],
        axis=1
    )

    # --------------------------------------------------
    # 5. ALIGN FEATURES WITH SCALER (MOST IMPORTANT FIX)
    # --------------------------------------------------
    expected_features = list(scaler.feature_names_in_)

    # Add missing columns (if any)
    for col in expected_features:
        if col not in processed_input.columns:
            processed_input[col] = 0

    # Reorder columns exactly as training
    processed_input = processed_input[expected_features]

   # --------------------------------------------------
# 6. FINAL ALIGNMENT WITH MODEL (CRITICAL FIX)
# --------------------------------------------------

# Get features expected by the model
expected_model_features = model.n_features_in_

# Convert scaled input to DataFrame to control columns
scaled_df = pd.DataFrame(
    scaled_input,
    columns=scaler.feature_names_in_
)

# If model was trained on fewer features than scaler
if scaled_df.shape[1] != expected_model_features:
    scaled_df = scaled_df.iloc[:, :expected_model_features]

# Convert back to numpy for prediction
final_input = scaled_df.values

# Predict
prediction = model.predict(final_input)[0]
