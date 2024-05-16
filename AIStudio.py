import streamlit as st
import pandas as pd
import base64
import numpy as np

# Function to generate acquisition dataset
def generate_acquisition_data(num_samples=1000):
    data = {
        'CustomerID': np.arange(1, num_samples + 1),
        'Age': np.random.randint(18, 70, num_samples),
        'Income': np.random.randint(20000, 100000, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], num_samples),
        'CreditScore': np.random.randint(300, 850, num_samples)
    }
    return pd.DataFrame(data)

# Function to generate activation dataset
def generate_activation_data(num_samples=1000):
    data = {
        'CustomerID': np.arange(1, num_samples + 1),
        'AccountCreatedDate': pd.date_range(start='2023-01-01', periods=num_samples, freq='D'),
        'InitialDeposit': np.random.randint(100, 5000, num_samples),
        'FirstTransactionAmount': np.random.randint(10, 1000, num_samples),
        'CardActivated': np.random.choice([True, False], num_samples),
        'OnlineBankingActivated': np.random.choice([True, False], num_samples)
    }
    return pd.DataFrame(data)

# Function to generate engagement dataset
def generate_engagement_data(num_samples=1000):
    data = {
        'CustomerID': np.arange(1, num_samples + 1),
        'TotalTransactions': np.random.randint(0, 100, num_samples),
        'TotalSpent': np.random.randint(0, 10000, num_samples),
        'CustomerSupportCalls': np.random.randint(0, 10, num_samples),
        'ServiceUsage': np.random.randint(0, 100, num_samples),
        'LastLoginDate': pd.date_range(start='2023-01-01', periods=num_samples, freq='D')
    }
    return pd.DataFrame(data)

# Function to generate retention dataset
def generate_retention_data(num_samples=1000):
    data = {
        'CustomerID': np.arange(1, num_samples + 1),
        'AccountAgeMonths': np.random.randint(1, 120, num_samples),
        'LastTransactionDate': pd.date_range(start='2023-01-01', periods=num_samples, freq='D'),
        'Churned': np.random.choice([True, False], num_samples),
        'TotalSpentLastYear': np.random.randint(0, 5000, num_samples),
        'CustomerSatisfactionScore': np.random.randint(1, 10, num_samples)
    }
    return pd.DataFrame(data)

# Generating and saving datasets as CSV files
acquisition_data = generate_acquisition_data()
activation_data = generate_activation_data()
engagement_data = generate_engagement_data()
retention_data = generate_retention_data()

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
        sample_file = {
            "Acquisition": acquisition_data,
            "Activation": activation_data,
            "Engagement": engagement_data,
            "Retention": retention_data
        }


        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode("utf-8")
        csv = convert_df(sample_file)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="large_df.csv",
            mime="text/csv",
        )

        
        def get_table_download_link(df):
            """Generates a link allowing the data in a given panda dataframe to be downloaded
            in:  dataframe
            out: href string
            """
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
        st.markdown(get_table_download_link(acquisition_data), unsafe_allow_html=True)

        
        # st.markdown(f'<a href="data:file/csv;base64,{b64}" download="acquisition_data.csv">Download csv file</a>')
        
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
