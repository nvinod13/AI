import streamlit as st
import pandas as pd
import base64
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to generate acquisition dataset
def generate_acquisition_data(num_samples=1000):
    data = {
        'CustomerID': np.arange(1, num_samples + 1),
        'Age': np.random.randint(18, 70, num_samples),
        'Income': np.random.randint(20000, 100000, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], num_samples),
        'CreditScore': np.random.randint(300, 850, num_samples),
        'LikelihoodToConvert': np.random.rand(num_samples)
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
        csv = convert_df(acquisition_data)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="large_df.csv",
            mime="text/csv",
        )

        # Acquisition
        # 1. Data Preparation and Feature Engineering
        def prepare_acquisition_data(df):
            # Assuming 'LikelihoodToConvert' is the target variable
            X = df.drop(columns=['CustomerID', 'LikelihoodToConvert'])
            y = df['LikelihoodToConvert']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, y_train, y_test, scaler

        # 2. Model Training and Prediction
        from sklearn.linear_model import LinearRegression
        def train_acquisition_model(X_train, y_train):
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model
        
        def predict_acquisition(model, X_test):
            predictions = model.predict(X_test)
            return prediction

        # 3. Model Validation
        from sklearn.metrics import mean_squared_error, r2_score
        def validate_acquisition_model(y_test, predictions):
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            return mse, r2

        # 4 Model Fine-tuning
        from sklearn.model_selection import GridSearchCV
        def fine_tune_acquisition_model(X_train, y_train):
            model = LinearRegression()
            param_grid = {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            return best_model
    
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
