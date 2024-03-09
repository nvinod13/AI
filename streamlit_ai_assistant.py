import pandas as pd
from transformers import pipeline

# Initialize GPT-Neo pipeline
gpt_neo = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# Sample dataset
data = {
    "Date": ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08"],
    "Description": ["Whole Foods Market", "Netflix Subscription", "Starbucks", "Electricity Bill", "AMC Theaters", "Local Water Utility", "McDonald's", "Trader Joe's"],
    "Category": ["Groceries", "Entertainment", "Dining Out", "Utilities", "Entertainment", "Utilities", "Dining Out", "Groceries"],
    "Amount ($)": [95.20, 13.99, 5.75, 60.00, 25.00, 30.00, 12.50, 55.00]
}
df = pd.DataFrame(data)

def search_dataset_for_category(category):
    # Sum amounts for the specified category
    category_sum = df[df['Category'].str.contains(category, case=False, na=False)]['Amount ($)'].sum()
    if category_sum > 0:
        return f"Your total spending in {category} is ${category_sum:.2f}."
    else:
        return "It seems like there were no expenses in that category."

def generate_response(prompt):
    # Use GPT-Neo to get an understanding of the query (simulated for this example)
    interpreted_query = "Groceries"  # Placeholder for the model's output
    
    # Generate a response based on the interpreted query
    response = search_dataset_for_category(interpreted_query)
    
    return response

# Example prompt from the user
prompt = "How much did I spend on groceries last month?"

# Generate response based on the prompt
response = generate_response(prompt)

print(response)
