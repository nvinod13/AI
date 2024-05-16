import streamlit as st
import pandas as pd

# Function to load a CSV file
def load_csv(file):
    return pd.read_csv(file)

# Function to save a dataframe to CSV
def save_csv(df, filename):
    df.to_csv(filename, index=False)

# Main function to run the Streamlit app
def main():
    st.title("Banking Use Case Model Library")

    # Model Library Screen
    st.sidebar.title("Model Library")
    use_case = st.sidebar.selectbox("Choose a use case", ["Acquisition", "Activation", "Engagement", "Retention"])
    
    if use_case:
        st.header(f"{use_case} Use Case")

        # Display use case description
        descriptions = {
            "Acquisition": "Predict the likelihood of a potential customer becoming a customer.",
            "Activation": "Determine if a new customer will activate their card.",
            "Engagement": "Estimate the total spending of an engaged customer.",
            "Retention": "Predict if an existing customer will churn."
        }
        st.write(descriptions[use_case])
        
        # Option to download a sample dataset
        st.subheader("Download Sample Dataset")
        sample_files = {
            "Acquisition": "acquisition_data.csv",
            "Activation": "activation_data.csv",
            "Engagement": "engagement_data.csv",
            "Retention": "retention_data.csv"
        }
        sample_file_name = sample_files[use_case]
        sample_file_path = f"/mnt/data/{sample_file_name}"
        st.download_button("Download Sample Dataset", sample_file_path)
        
        # Option to upload a dataset
        st.subheader("Upload Your Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            data = load_csv(uploaded_file)
            st.write("Uploaded Dataset:")
            st.write(data.head())
            
            # Run model on uploaded data
            st.subheader("Run Model")
            if st.button("Run Model"):
                if use_case == "Acquisition":
                    X_train, X_test, y_train, y_test, scaler = prepare_acquisition_data(data)
                    model = train_acquisition_model(X_train, y_train)
                    predictions = predict_acquisition(model, X_test)
                    mse, r2 = validate_acquisition_model(y_test, predictions)
                    st.write(f"Mean Squared Error: {mse}")
                    st.write(f"R-squared: {r2}")
                    output_data = create_acquisition_output_dataset(data, model, scaler)
                
                elif use_case == "Activation":
                    X_train, X_test, y_train, y_test, scaler = prepare_activation_data(data)
                    model = train_activation_model(X_train, y_train)
                    predictions = predict_activation(model, X_test)
                    accuracy, conf_matrix, class_report = validate_activation_model(y_test, predictions)
                    st.write(f"Accuracy: {accuracy}")
                    st.write("Confusion Matrix:")
                    st.write(conf_matrix)
                    st.write("Classification Report:")
                    st.write(class_report)
                    output_data = create_activation_output_dataset(data, model, scaler)
                
                elif use_case == "Engagement":
                    X_train, X_test, y_train, y_test, scaler = prepare_engagement_data(data)
                    model = train_engagement_model(X_train, y_train)
                    predictions = predict_engagement(model, X_test)
                    mse, r2 = validate_engagement_model(y_test, predictions)
                    st.write(f"Mean Squared Error: {mse}")
                    st.write(f"R-squared: {r2}")
                    output_data = create_engagement_output_dataset(data, model, scaler)
                
                elif use_case == "Retention":
                    X_train, X_test, y_train, y_test, scaler = prepare_retention_data(data)
                    model = train_retention_model(X_train, y_train)
                    predictions = predict_retention(model, X_test)
                    accuracy, conf_matrix, class_report = validate_retention_model(y_test, predictions)
                    st.write(f"Accuracy: {accuracy}")
                    st.write("Confusion Matrix:")
                    st.write(conf_matrix)
                    st.write("Classification Report:")
                    st.write(class_report)
                    output_data = create_retention_output_dataset(data, model, scaler)
                
                st.write("Output Dataset with Predictions:")
                st.write(output_data.head())
                
                # Option to download the output dataset
                output_file_name = f"{use_case.lower()}_output_data.csv"
                save_csv(output_data, output_file_name)
                st.download_button("Download Output Dataset", output_file_name)

if __name__ == "__main__":
    main()
