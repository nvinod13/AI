import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data for the initial dataset
def generate_sample_data():
    data = {
        "Transaction ID": [f"TXN00{i}" for i in range(1, 26)],
        "Transaction Date and Time": pd.date_range(start='2024-06-24 10:00:00', periods=25, freq='H'),
        "Transaction Amount": np.random.randint(100, 5000, size=25),
        "Transaction Type": np.random.choice(["Remittance", "Bill Payment", "Prepaid Top-Up", "Currency Exchange", "B2B Remittance", "Government Aid"], size=25),
        "Transaction Status": np.random.choice(["Completed", "Pending"], size=25),
        "Processing Time": np.random.randint(1, 15, size=25),
        "Customer ID": [f"CUST00{i}" for i in range(1, 26)],
        "Customer Demographics": np.random.choice(["Age: 30, M, NY, USA", "Age: 25, F, CA, USA", "Age: 40, M, TX, USA"], size=25),
        "Customer Transaction History": np.random.randint(1, 20, size=25),
        "Payment Method": np.random.choice(["Bank Transfer", "Credit Card", "Mobile Wallet"], size=25),
        "In-Person Transactions": np.random.choice(["Yes", "No"], size=25),
        "Digital Transactions": np.random.choice(["Yes", "No"], size=25),
        "Fee Details": np.random.uniform(1.0, 15.0, size=25),
        "Promotion Details": np.random.choice(["Promo10%", "Promo5%", "None"], size=25),
        "Business Client ID": [f"BUS00{i}" if i % 5 == 0 else "N/A" for i in range(1, 26)],
        "Business Client Details": np.random.choice(["Tech Company", "Retailer", "N/A"], size=25),
        "Payment Volume": np.random.randint(1000, 100000, size=25),
        "Foreign Exchange Rates": np.random.uniform(0.5, 1.5, size=25),
        "Hedging Transactions": np.random.randint(100, 10000, size=25),
        "Recipient ID": [f"REC00{i}" if i % 3 == 0 else "N/A" for i in range(1, 26)],
        "Recipient Verification Details": np.random.choice(["Fingerprint", "Facial Recognition", "N/A"], size=25),
        "Disbursement Schedule": np.random.choice(["Monthly", "Quarterly", "N/A"], size=25),
        "Aid and Grant Details": np.random.choice(["Aid for Education", "Healthcare Aid", "N/A"], size=25),
        "Utility Provider ID": [f"UTIL00{i}" if i % 4 == 0 else "N/A" for i in range(1, 26)],
        "Bill Details": np.random.choice(["$100, Due 2024-06-30", "$75, Due 2024-07-05", "N/A"], size=25),
        "Payment Confirmation": np.random.choice(["Confirmed", "N/A"], size=25),
        "Prepaid Card ID": [f"CARD00{i}" if i % 2 == 0 else "N/A" for i in range(1, 26)],
        "Top-Up Amount": np.random.randint(10, 100, size=25),
        "Top-Up Frequency": np.random.choice(["Monthly", "Weekly", "Bi-Weekly"], size=25),
        "Currency Pair": np.random.choice(["USD/EUR", "USD/JPY", "GBP/USD"], size=25),
        "Exchange Rate": np.random.uniform(0.8, 1.5, size=25),
        "Exchange Volume": np.random.randint(100, 5000, size=25),
        "Staffing Levels": np.random.randint(3, 15, size=25),
        "Error Rates": np.random.randint(0, 5, size=25),
        "System Performance Metrics": np.random.uniform(99.5, 100.0, size=25),
        "Customer Feedback": np.random.choice(["4 stars", "5 stars", "3.5 stars"], size=25),
        "Support Interaction Records": np.random.randint(1, 5, size=25),
        "Loyalty Program Participation": np.random.choice(["Yes", "No"], size=25),
        "Compliance Check Results": np.random.choice(["Compliant", "Pending"], size=25),
        "Fraud Alerts": np.random.choice(["None", "Alert"], size=25),
        "Regulatory Changes": np.random.choice(["N/A", "Updated AML rules"], size=25),
        "Promotion ID": [f"PROMO00{i}" for i in range(1, 26)],
        "Promotion Effectiveness": np.random.choice(["20% uptake", "30% uptake", "15% uptake"], size=25),
        "Referral Program Participation": np.random.randint(1, 10, size=25),
        "Historical Data Trends": np.random.choice(["Increasing volume", "Stable trends"], size=25),
        "External Market Data": np.random.choice(["Stable exchange rates", "Increasing utility costs"], size=25),
    }
    return pd.DataFrame(data)

# Load sample data
sample_data = generate_sample_data()

# App Layout
st.title("Remittance AI Dashboard")

# Tabs
tabs = st.tabs(["View Sample Data", "CxO Agent", "Consumer Agent", "Employee Agent", "Partner Agent"])

# View Sample Data Tab
with tabs[0]:
    st.header("Sample Data")
    st.dataframe(sample_data.head(25))

# CxO Concierge Tab
with tabs[1]:
    st.header("CxO Agent")
    query = st.text_input("How many FX transactions can we expect next week in the TH-MY corridor?", key="cxo")
    if st.button("Search", key="cxo_button"):
        if query.lower() == "how many fx transactions can we expect next week in the th-my corridor?":
            st.subheader("Weekly Forecast of FX Transactions in TH-MY Corridor")
            weeks = ["Week -4", "Week -3", "Week -2", "Week -1", "Next Week"]
            values = np.random.randint(100, 200, size=5)
            plt.figure(figsize=(10, 6))
            plt.plot(weeks, values, marker='o')
            plt.title("Weekly FX Transactions Forecast")
            plt.xlabel("Week")
            plt.ylabel("Number of Transactions")
            plt.grid(True)
            st.pyplot(plt)

# Consumer App Tab
with tabs[2]:
    st.header("Consumer Agent")
    query = st.text_input("When are FX rates from TH-MY likely to be lowest over the next 4 weeks?", key="consumer")
    if st.button("Search", key="consumer_button"):
        if query.lower() == "when are fx rates from th-my likely to be lowest over the next 4 weeks?":
            st.subheader("Weekly Forecast of FX Rates in TH-MY Corridor")
            weeks = ["Week -4", "Week -3", "Week -2", "Week -1", "Next Week", "Week +1", "Week +2", "Week +3"]
            values = np.random.uniform(0.25, 0.75, size=8)
            plt.figure(figsize=(10, 6))
            plt.plot(weeks, values, marker='o')
            plt.title("Weekly FX Rates Forecast")
            plt.xlabel("Week")
            plt.ylabel("FX Rate")
            plt.grid(True)
            st.pyplot(plt)

# Employee App Tab
with tabs[3]:
    st.header("Employee Agent")
    query = st.text_input("How many walk-ins can I expect in Pagoda Street Branch after 2pm next Saturday?", key="employee")
    if st.button("Search", key="employee_button"):
        if query.lower() == "how many walk-ins can i expect in pagoda street branch after 2pm next saturday?":
            st.subheader("Daily Forecast of Customer Walk-Ins at Pagoda Street Branch")
            days = ["Saturday -4", "Saturday -3", "Saturday -2", "Saturday -1", "Next Saturday"]
            total_walkins = np.random.randint(50, 100, size=5)
            walkins_after_2pm = np.random.randint(20, 50, size=5)
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].plot(days, total_walkins, marker='o', label="Total Walk-Ins")
            ax[0].set_title("Total Customer Walk-Ins")
            ax[0].set_xlabel("Day")
            ax[0].set_ylabel("Number of Walk-Ins")
            ax[0].grid(True)
            ax[0].legend()
            ax[1].plot(days, walkins_after_2pm, marker='o', label="Walk-Ins After 2 PM")
            ax[1].set_title("Customer Walk-Ins After 2 PM")
            ax[1].set_xlabel("Day")
            ax[1].set_ylabel("Number of Walk-Ins")
            ax[1].grid(True)
            ax[1].legend()
            st.pyplot(fig)

# Agent App Tab
with tabs[4]:
    st.header("Partner Agent")
    query = st.text_input("How far am I from meeting my customer acquisition targets for this quarter?", key="agent")
    if st.button("Search", key="agent_button"):
        if query.lower() == "how far am i from meeting my customer acquisition targets for this quarter?":
            st.subheader("Customer Acquisition Forecast")
            weeks = ["Week +1", "Week +2", "Week +3", "Week +4"]
            customers_acquired = [20, 15, 12, 10]
            target = 80
            variance = target - sum(customers_acquired)
            plt.figure(figsize=(10, 6))
            plt.plot(weeks, customers_acquired, marker='o')
            plt.title("Weekly Customer Acquisition Forecast")
            plt.xlabel("Week")
            plt.ylabel("Number of Customers Acquired")
            plt.grid(True)
            st.pyplot(plt)
            st.write(f"Variance from target of 80 customers: {variance}")
