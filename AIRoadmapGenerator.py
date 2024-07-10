import streamlit as st
import pandas as pd
import datetime
from pptx import Presentation
from pptx.util import Inches

# Define the use cases
use_cases = [
    {"Category": "Product", "Use Case": "Personalized product recommendations", "Audience": "Client Focus", "Impact": "Revenue Increase"},
    {"Category": "Product", "Use Case": "Dynamic pricing optimization", "Audience": "Client Focus", "Impact": "Revenue Increase"},
    {"Category": "Product", "Use Case": "Product lifecycle management", "Audience": "Employee Focus", "Impact": "Productivity Improvements"},
    {"Category": "Product", "Use Case": "Customer churn prediction", "Audience": "Client Focus", "Impact": "Revenue Increase"},
    # Add other use cases similarly...
]

# Convert the use cases to a DataFrame
use_cases_df = pd.DataFrame(use_cases)

# Initialize session state
if "assigned_use_cases" not in st.session_state:
    st.session_state.assigned_use_cases = []

# Define functions for PPT generation
def generate_ppt(assigned_use_cases):
    prs = Presentation()
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "AI Roadmap"

    left = Inches(1)
    top = Inches(1.5)
    width = Inches(8.5)
    height = Inches(0.5)

    for use_case in assigned_use_cases:
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        p = tf.add_paragraph()
        p.text = f"{use_case['Use Case']} - {use_case['Category']} - {use_case['Audience']} - {use_case['Impact']} - {use_case['Importance']} - {use_case['Urgency']} - {use_case['Timeline']} - {use_case['Budget']}"
        top += Inches(0.5)
    
    return prs

# UI for Use Case Assignment
st.title("AI Roadmap Generator")

st.subheader("Assign Importance, Urgency, Timelines, and Budgets to Use Cases")

selected_use_case = st.selectbox("Select a Use Case", use_cases_df["Use Case"])
importance = st.slider("Importance", 1, 10)
urgency = st.slider("Urgency", 1, 10)
timeline = st.date_input("Timeline", datetime.date.today())
budget = st.number_input("Budget", min_value=0)

if st.button("Assign Use Case"):
    assigned_use_case = {
        "Use Case": selected_use_case,
        "Category": use_cases_df[use_cases_df["Use Case"] == selected_use_case]["Category"].values[0],
        "Audience": use_cases_df[use_cases_df["Use Case"] == selected_use_case]["Audience"].values[0],
        "Impact": use_cases_df[use_cases_df["Use Case"] == selected_use_case]["Impact"].values[0],
        "Importance": importance,
        "Urgency": urgency,
        "Timeline": timeline,
        "Budget": budget
    }
    st.session_state.assigned_use_cases.append(assigned_use_case)

st.subheader("Assigned Use Cases")
st.write(pd.DataFrame(st.session_state.assigned_use_cases))

# UI for Gantt Chart Visualization (Mockup)
st.subheader("Gantt Chart Visualization (Mockup)")
# Note: In a real implementation, you would use a library to generate a Gantt chart
st.write("Gantt Chart visualization would appear here.")

# Generate PowerPoint
if st.button("Generate PowerPoint"):
    prs = generate_ppt(st.session_state.assigned_use_cases)
    prs.save('/tmp/AI_Roadmap.pptx')
    with open('/tmp/AI_Roadmap.pptx', "rb") as ppt_file:
        st.download_button(
            label="Download PowerPoint",
            data=ppt_file,
            file_name="AI_Roadmap.pptx"
        )

st.sidebar.subheader("Budget Tracker")
total_budget = st.sidebar.number_input("Total Budget", min_value=0)
assigned_budget = sum(use_case["Budget"] for use_case in st.session_state.assigned_use_cases)
remaining_budget = total_budget - assigned_budget

st.sidebar.write(f"Assigned Budget: {assigned_budget}")
st.sidebar.write(f"Remaining Budget: {remaining_budget}")

if remaining_budget < 0:
    st.sidebar.error("Budget exceeded! Adjust your use cases or increase the total budget.")
else:
    st.sidebar.success("Budget is within the limit.")
