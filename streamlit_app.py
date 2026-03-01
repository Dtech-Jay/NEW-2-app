import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the trained model and preprocessing objects ---
model = joblib.load('linear_regression_model.joblib')
scaler = joblib.load('standard_scaler.joblib')
encoder = joblib.load('one_hot_encoder.joblib')

# --- 2. Streamlit App Title and Description ---
st.title('Total Revenue Prediction App')
st.write('Enter the product and customer details to predict the total revenue.')

# --- 3. Create input widgets for features ---
st.header('Product Details')
price = st.number_input('Price', min_value=0.0, value=100.0, step=1.0)
discount_percent = st.number_input('Discount Percent', min_value=0, max_value=100, value=10, step=1)
quantity_sold = st.number_input('Quantity Sold', min_value=1, value=1, step=1)
rating = st.slider('Rating', min_value=1.0, max_value=5.0, value=4.0, step=0.1)
review_count = st.number_input('Review Count', min_value=0, value=100, step=1)

product_category_options = encoder.categories_[0]
product_category = st.selectbox('Product Category', product_category_options)

st.header('Customer Details')
customer_region_options = encoder.categories_[1]
customer_region = st.selectbox('Customer Region', customer_region_options)

payment_method_options = encoder.categories_[2]
payment_method = st.selectbox('Payment Method', payment_method_options)

# --- 4. Prediction Button ---
if st.button('Predict Total Revenue'):
    # --- 5. Preprocess user input ---
    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'price': [price],
        'discount_percent': [discount_percent],
        'quantity_sold': [quantity_sold],
        'rating': [rating],
        'review_count': [review_count],
        'discounted_price': [0.0], # Placeholder, will be calculated later or estimated
        # total_revenue is the target, not an input
        'product_category': [product_category],
        'customer_region': [customer_region],
        'payment_method': [payment_method]
    })

    # Calculate discounted_price (since it's a feature in the model)
    # Assuming discounted_price = price * (1 - discount_percent/100)
    input_data['discounted_price'] = input_data['price'] * (1 - input_data['discount_percent'] / 100)

    # Identify categorical and numerical columns for preprocessing
    categorical_cols = ['product_category', 'customer_region', 'payment_method']
    numerical_cols = ['price', 'discount_percent', 'quantity_sold', 'rating', 'review_count', 'discounted_price']

    # One-hot encode categorical features
    encoded_features = encoder.transform(input_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    # Concatenate numerical and encoded categorical features
    processed_input = pd.concat([
        input_data[numerical_cols].reset_index(drop=True),
        encoded_df
    ], axis=1)

    # The full list of columns expected by the model after all preprocessing (excluding 'order_id', 'product_id', 'order_date', 'total_revenue')
    # Based on the X_train in kernel state, these are the columns:
    model_features = ['price', 'discount_percent', 'quantity_sold', 'rating', 'review_count', 'discounted_price',
                      'product_category_Beauty', 'product_category_Books', 'product_category_Electronics', 'product_category_Fashion', 'product_category_Home & Kitchen', 'product_category_Sports',
                      'customer_region_Asia', 'customer_region_Europe', 'customer_region_Middle East', 'customer_region_North America',
                      'payment_method_Cash on Delivery', 'payment_method_Credit Card', 'payment_method_Debit Card', 'payment_method_UPI', 'payment_method_Wallet']
    
    # Reorder columns to match the training data's feature order
    processed_input = processed_input[model_features]

    # Scale numerical features (note: all features including encoded ones were scaled during training)
    # The scaler was fitted on a DataFrame that included both direct numerical and one-hot encoded columns.
    # So we apply scaler to the entire processed_input dataframe.
    scaled_input = scaler.transform(processed_input)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]

    st.success(f'Predicted Total Revenue: {prediction:.2f}')
