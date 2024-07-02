import streamlit as st
import openai

from gpt_researcher import GPTResearcher
import asyncio
import os
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
os.environ["OPENAI_API_KEY"] = "sk-proj-pakReqzN0s9J7cUODfZCT3BlbkFJK2tkGOSvuuORWBSv1c1j"
os.environ["TAVILY_API_KEY"] = "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"
# It is best to define global constants at the top of your script
QUERY = "AI and digital initiatives of Al Rajhi Bank"
REPORT_TYPE = "research_report"

# Set your OpenAI API key
openai.api_key = 'sk-proj-pakReqzN0s9J7cUODfZCT3BlbkFJK2tkGOSvuuORWBSv1c1j'

def query_llm(prompt):
    async def fetch_report(query, report_type):
        """
        Fetch a research report based on the provided query and report type.
        """
        researcher = GPTResearcher(query=query, report_type=report_type, config_path=None)
        await researcher.conduct_research()
        report = await researcher.write_report()
        return report

    async def generate_research_report():
        """
        This is a sample script that executes an async main function to run a research report.
        """
        report = await fetch_report(QUERY, REPORT_TYPE)
        print(report)


def main():
    st.title("Research Agent")

    # Text input for user's query
    user_input = st.text_input("Input", "Digital and AI initiatives of Al Rajhi Bank")
    st.caption("Digital and AI initiatives of Al Rajhi Bank")

    # Search button
    if st.button("Search"):
        if user_input:
            with st.spinner("Querying LLM..."):
                output = query_llm(user_input)
                st.success("Query Complete!")
                st.write(output)
        else:
            st.error("Please enter a valid input")

if __name__ == "__main__":
    main()
