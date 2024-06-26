import streamlit as st
import pandas as pd
from faker import Faker
fake = Faker()

# Placeholder for global data storage in multi-page app scenarios
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()

# Function to generate dummy data based on system selections
def generate_dummy_data(system_selections, num_rows=100):
    data = {
        'CustomerID': [fake.unique.random_int(min=10000, max=99999) for _ in range(num_rows)],
        'Name': [fake.name() for _ in range(num_rows)],
        'Email': [fake.email() for _ in range(num_rows)],
        'Phone Number': [fake.phone_number() for _ in range(num_rows)],
        'Transaction Date': [fake.date_between(start_date='-1y', end_date='today') for _ in range(num_rows)],
        'Transaction Amount': [fake.random_number(digits=5) for _ in range(num_rows)],
        'Account Type': [fake.random_element(elements=('Savings', 'Checking', 'Loan')) for _ in range(num_rows)]
    }
    
    # You can customize the data further based on system selections if necessary
    # For example, add specific fields or modify existing ones based on the selected systems
    
    return pd.DataFrame(data)
    
# Define GUI for each engine
def onboarding_engine():
   st.header("Onboarding Engine")
# Platform/system selection
core_banking_platform = st.selectbox("Select Core Banking Platform", ["Temenos T24", "Oracle FLEXCUBE", "SAP Banking", "Infosys Finacle", "FIS Profile"])
cards_management_system = st.selectbox("Select Cards Management System", ["System 1", "System 2", "System 3", "System 4", "System 5"])
loan_origination_system = st.selectbox("Select Loan Origination System", ["LOS 1", "LOS 2", "LOS 3", "LOS 4", "LOS 5"])
internet_banking_platform = st.selectbox("Select Internet Banking Platform", ["IBP 1", "IBP 2", "IBP 3", "IBP 4", "IBP 5"])
data_warehouse = st.selectbox("Select Data Warehouse", ["DW 1", "DW 2", "DW 3", "DW 4", "DW 5"])
    
# User authentication
userid = st.text_input("UserID")
password = st.text_input("Password", type="password")
    
# Database selection
database = st.selectbox("Select Database", ["Database 1", "Database 2", "Database 3"])
    
# Date picker for timeframe
start_date, end_date = st.select_slider("Select a date range",
        options=pd.date_range("2020-01-01", "2023-12-31").tolist(),
        value=(pd.Timestamp("2020-01-01"), pd.Timestamp("2023-12-31"))
    )

# Button to generate dummy data
if st.button('Generate Dummy Data'):
        system_selections = {
            'core_banking_platform': core_banking_platform,
            'cards_management_system': cards_management_system,
            'loan_origination_system': loan_origination_system,
            'internet_banking_platform': internet_banking_platform,
            'data_warehouse': data_warehouse,
            'database': database
        }
# Generate dummy data based on selected systems
        st.session_state['data'] = generate_dummy_data(system_selections, num_rows=100)
        st.success('Dummy data generated based on your selections.')

# Display generated data if available
if 'data' in st.session_state and not st.session_state['data'].empty:
        st.write("Generated Data Preview:")
        st.dataframe(st.session_state['data'])

def auto_mapping_engine():
    st.header("Auto-mapping Engine")

    if not st.session_state.get('data', pd.DataFrame()).empty:
        st.write("Source to Target Column Mappings:")
        
        # Example static mapping - in practice, this could be dynamic based on actual data columns
        mappings = {
            "Source Column": ["Card Number", "Phone Number", "Address", "Email", "Date"],
            "Target Column": ["card_num", "phone_num", "address", "email", "date"],
            "Reasoning": ["Unique identifier", "Contact info", "Location data", "Contact info", "Temporal data"]
        }
        df_mappings = pd.DataFrame(mappings)
        st.table(df_mappings)

        if st.button("Apply Auto-mapping"):
            st.session_state.data['mapping_applied'] = True  # Placeholder action
            st.success("Auto-mapping applied.")
    else:
        st.write("Please upload data in the Onboarding Engine first.")

def cleaning_engine():
    st.header("Cleaning Engine")
    
    if not st.session_state.get('data', pd.DataFrame()).empty:
        # Options for data transformations
        transformation_options = st.multiselect("Select Transformations to Apply",
                                                ["Remove Duplicates", "Handle Missing Values", "Normalize Data"])
        
        if st.button("Apply Transformations"):
            for option in transformation_options:
                # Placeholder for actual transformation logic
                st.session_state.data[f'{option}_applied'] = True
            st.success("Data transformations applied.")
        st.write("Transformed Data Preview:")
        st.dataframe(st.session_state.data)  # Display the modified DataFrame
    else:
        st.write("Please upload and map data before cleaning.")

def enrichment_engine():
    st.header("Enrichment Engine")
     
    if not st.session_state.get('data', pd.DataFrame()).empty:
        # Options for data enrichment
        enrichment_options = st.multiselect("Select Data Enrichment Options",
                                            ["Merchant Data with Tags", "Location Data with Population Density", "Mobile Device ID with Interests and Affluence"])
        
        if st.button("Apply Enrichment"):
            for option in enrichment_options:
                # Placeholder for actual enrichment logic
                st.session_state.data[f'{option}_enriched'] = True
            st.success("Data enrichment applied.")
        st.write("Enriched Data Preview:")
        st.dataframe(st.session_state.data)  # Display the modified DataFrame
    else:
        st.write("Please ensure data is uploaded, mapped, and cleaned before enrichment.")

# App layout with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Onboarding Engine", "Auto-mapping Engine", "Cleaning Engine", "Enrichment Engine"])

with tab1:
    onboarding_engine()

with tab2:
    auto_mapping_engine()

with tab3:
    cleaning_engine()

with tab4:
    enrichment_engine()
