import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os  # Import the os module

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model using a relative path
model_path = os.path.join(current_dir, "trained_model.joblib")
model = joblib.load(model_path)  # ✅ FIXED: Works on Streamlit Cloud


# Define required features and realistic value ranges for Iris dataset
required_features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

# Streamlit UI
st.title("🌸 Iris Flower Species Prediction")

# Choose input method
option = st.radio("Select Input Method:", ["📂 Upload CSV File", "✍️ Manual Input"])

def clean_and_impute(df):
    """ Cleans the dataframe by handling missing, negative, and out-of-range values. """
    df.replace(["error", "NA", "??", "none", "missing", ""], np.nan, inplace=True)

    # Convert to numeric and handle missing values
    for col in required_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Define realistic value ranges for the Iris dataset
    valid_ranges = {
        "SepalLengthCm": (4.3, 7.9),
        "SepalWidthCm": (2.0, 4.4),
        "PetalLengthCm": (1.0, 6.9),
        "PetalWidthCm": (0.1, 2.5)
    }

    # Impute invalid values (negative, out of range) with column mean
    for col in required_features:
        col_mean = df[col].mean()  # Compute mean ignoring NaNs
        df[col] = df[col].apply(lambda x: col_mean if x < 0 or x < valid_ranges[col][0] or x > valid_ranges[col][1] else x)

    df.fillna(df.mean(), inplace=True)  # Final NaN handling
    return df

if option == "📂 Upload CSV File":
    uploaded_file = st.file_uploader("📂 Upload CSV file for prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📌 **Raw Data Preview:**")
            st.dataframe(df.head())

            missing_features = [col for col in required_features if col not in df.columns]

            if missing_features:
                st.error(f"🚨 Missing Features: {missing_features}. Please check the uploaded file and include all required columns.")
            else:
                df = clean_and_impute(df)  # Clean the data

                # Make Predictions
                predictions = model.predict(df[required_features].to_numpy())  
                df["Predicted_Species"] = predictions  # Directly assign predictions

                # Display Predictions
                st.write("✅ **Predicted Results:**")
                st.dataframe(df)

                # Display Prediction Count Summary
                st.subheader("📊 Prediction Count Summary:")
                prediction_counts = df["Predicted_Species"].value_counts()

                for species, count in prediction_counts.items():
                    st.write(f"🌸 {species}: {count} predictions")

                # Generate Pie Chart
                fig = px.pie(
                    names=prediction_counts.index, 
                    values=prediction_counts.values, 
                    title="Distribution of Predicted Flower Types",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                st.plotly_chart(fig)

                # Downloadable Results
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", csv, "predicted_results.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}. Please try again with a valid file.")

elif option == "✍️ Manual Input":
    st.subheader("📝 Enter Values Manually:")

    sepal_length = st.number_input("Sepal Length (cm)", step=0.1, value=5.0)
    sepal_width = st.number_input("Sepal Width (cm)", step=0.1, value=3.0)
    petal_length = st.number_input("Petal Length (cm)", step=0.1, value=1.5)
    petal_width = st.number_input("Petal Width (cm)", step=0.1, value=0.2)

    if st.button("🔍 Predict"):
        try:
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

            # Clean the input (handle negatives & out-of-range values)
            df_manual = pd.DataFrame(input_data, columns=required_features)
            df_manual = clean_and_impute(df_manual)

            prediction = model.predict(df_manual.to_numpy())[0]  # Direct prediction
            st.success(f"🌸 Predicted Flower Species: **{prediction}**")  # Display directly

        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}. Please enter valid values.")
