from transformers import pipeline
import pandas as pd
import numpy as np

# Initialize the GPT-Neo pipeline
gpt_neo = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# Sample dataset
np.random.seed(0)
data = {
    'Date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
    'Category': np.random.choice(['Groceries', 'Entertainment', 'Utilities'], size=10),
    'Amount ($)': np.round(np.random.uniform(10, 100, size=10), 2),
    'Time of Day': np.random.choice(['Morning', 'Afternoon', 'Evening'], size=10),
    'State': np.random.choice(['State1', 'State2', 'State3'], size=10),
}
df = pd.DataFrame(data)

# Functions to handle queries
def get_summary_by_column(column_name):
    if column_name in df.columns:
        return df.groupby(column_name)['Amount ($)'].describe()
    else:
        return f"Column {column_name} not found."

def get_spending_metrics():
    metrics = {
        'Highest Spend': df['Amount ($)'].max(),
        'Lowest Spend': df['Amount ($)'].min(),
        'Average Spend': df['Amount ($)'].mean(),
        'Median Spend': df['Amount ($)'].median(),
    }
    return metrics

def get_grouped_spends(col1, col2):
    if col1 in df.columns and col2 in df.columns:
        return df.groupby([col1, col2])['Amount ($)'].sum().reset_index(name='Total Spent')
    else:
        return "One or both columns not found."

# Simulate GPT-Neo Interpretation (replace this with actual GPT-Neo call for real use)
def interpret_query(query):
    # This is a placeholder for the GPT-Neo interpretation.
    # You would replace this logic with actual calls to gpt_neo() function
    # and then parse its output to understand the user's intent.
    if "summary" in query:
        return "summary", "Category"
    elif "metrics" in query:
        return "metrics", None
    elif "grouped" in query:
        return "grouped", ("Category", "State")
    else:
        return "unknown", None

# Example query processing
def process_query(query):
    interpreted_action, details = interpret_query(query)
    
    if interpreted_action == "summary":
        column_name = details
        result = get_summary_by_column(column_name)
    elif interpreted_action == "metrics":
        result = get_spending_metrics()
    elif interpreted_action == "grouped":
        col1, col2 = details
        result = get_grouped_spends(col1, col2)
    else:
        result = "Sorry, I couldn't understand your request."
    
    return result

# Example Usage
query = "I want a summary by Category"
result = process_query(query)
print(result)
