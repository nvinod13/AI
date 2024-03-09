import streamlit as st
import pandas as pd

# Sample dataset
data = {
    "Date": ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"],
    "Description": ["Whole Foods Market", "Netflix Subscription", "Starbucks", "Electricity Bill", "AMC Theaters", "Local Water Utility", "McDonald's", "Trader Joe's"],
    "Category": ["Groceries", "Entertainment", "Dining Out", "Utilities", "Entertainment", "Utilities", "Dining Out", "Groceries"],
    "Amount ($)": [95.20, 13.99, 5.75, 60.00, 25.00, 30.00, 12.50, 55.00]
}

df = pd.DataFrame(data)

def ai_assistant_improved(prompt):
    prompt = prompt.lower().replace("?", "")
    if "total spending" in prompt:
        total_spending = df["Amount ($)"].sum()
        return f"Your total spending is ${total_spending:.2f}."
    elif "spending in" in prompt:
        category = prompt.split("spending in ")[1].capitalize()
        if category in df["Category"].str.capitalize().unique():
            category_spending = df[df["Category"].str.capitalize() == category]["Amount ($)"].sum()
            return f"Your spending in {category} is ${category_spending:.2f}."
        else:
            return f"Category '{category}' not found in the dataset."
    else:
        return "Sorry, I didn't understand that. Please ask about your total spending or spending in a specific category."

# Streamlit app
st.title('AI Finance Assistant')

user_prompt = st.text_input("Ask about your spending:")

if user_prompt:
    response = ai_assistant_improved(user_prompt)
    st.write(response)
