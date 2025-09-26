# backend.py
import os
import json
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool

# ---------------------------
# Load OpenAI API key
# ---------------------------
import streamlit as st

# Get the key from Streamlit Secrets
api_key = st.secrets["openai"]["api_key"]

# Initialize the OpenAI LLM with the key
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=api_key)

load_dotenv()
#api_key = os.environ.get("OPENAI_API_KEY")

# ---------------------------
# Load Walmart dataset
# ---------------------------
df = pd.read_csv("Walmart.csv")
df['unit_price'] = df['unit_price'].str.replace('$', '').astype(float)
df['Sales'] = df['unit_price'] * df['quantity']

# ---------------------------
# LLM setup
# ---------------------------
#llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=api_key)

# ---------------------------
# Tools for AI Agent
# ---------------------------
def summary_tool(query: str) -> str:
    numeric_cols = df.select_dtypes(include='number').columns
    desc = df[numeric_cols].describe().to_string()
    insights = [f"{col} has mean {df[col].mean():.2f} and std {df[col].std():.2f}" for col in numeric_cols]

    llm_summary = llm.invoke(f"""
    Here are the statistics for Walmart dataset:
    {desc}

    Insights:
    {"; ".join(insights)}

    Please write a concise, business-oriented summary in 3–4 sentences.
    """)
    return f"Summary stats:\n{desc}\n\nAI-generated summary:\n{llm_summary.content}\n\nInsights:\n" + "\n".join(insights)

def missing_tool(query: str) -> str:
    return f"Missing values per column:\n{df.isnull().sum().to_dict()}"

def correlation_tool(query: str) -> str:
    numeric_cols = df.select_dtypes(include='number').columns
    corr = df[numeric_cols].corr()
    return f"Correlation matrix:\n{corr.to_dict()}"

def top_sales_tool(query: str) -> str:
    idx = df['Sales'].idxmax()
    return f"Invoice {df.loc[idx, 'invoice_id']} in Branch {df.loc[idx, 'Branch']} had the highest sales: {df.loc[idx, 'Sales']:.2f}"

def plots_tool(query: str) -> str:
    return "Plots are rendered in Streamlit frontend."

def query_tool(user_input: str) -> str:
    columns = list(df.columns)
    llm_response = llm.invoke(f"""
    You are a query generator. The Walmart dataset has these columns:
    {columns}

    User’s question: "{user_input}"

    Return ONLY:
    1. Python (pandas) query (assume dataframe is df)
    2. SQL query (assume table name = walmart)
    """)
    return llm_response.content

def general_question_tool(user_input: str) -> str:
    columns = list(df.columns)
    llm_response = llm.invoke(f"""
    You are a data analyst for the Walmart dataset with columns: {columns}

    User Question: "{user_input}"

    Provide:
    1. Python (pandas) code to answer the question (assume df as dataframe)
    2. SQL query (assume table name = walmart)
    3. Brief explanation in human language
    4. Provide output using the Walmart.csv dataset 

    Return in clear sections:
    Python:
    <code>

    SQL:
    <code>

    Explanation:
    <text>

    Answer:
    <text>
    """)
    return llm_response.content

# ---------------------------
# Convert to LangChain Tool objects
# ---------------------------
tools = [
    Tool(name="summary_tool", func=summary_tool, description="Gives summary statistics and numeric insights"),
    Tool(name="missing_tool", func=missing_tool, description="Shows missing values per column"),
    Tool(name="correlation_tool", func=correlation_tool, description="Shows correlation matrix"),
    Tool(name="top_sales_tool", func=top_sales_tool, description="Shows the invoice with highest sales"),
    Tool(name="plots_tool", func=plots_tool, description="Generates plots in frontend"),
    Tool(name="query_tool", func=query_tool, description="Generates Python + SQL queries for user questions"),
    Tool(name="general_question_tool", func=general_question_tool, description="Answer any general dataset question with Python + SQL + explanation + answer")
]

# ---------------------------
# Fixed-question menu
# ---------------------------
fixed_questions = {
    "1": "Show summary statistics",
    "2": "Show missing values",
    "3": "Show correlation matrix",
    "4": "Show invoice with highest sales",
    "5": "Generate all plots",
    "6": "Full analysis (all tools)",
    "7": "Query Generator in Python & SQL",
    "8": "General Questions with explanation and code"
}

# ---------------------------
# Answer query function
# ---------------------------
def answer_query(choice, user_input=None):
    if choice == "6":
        return {
            "output": json.dumps({
                "summary": summary_tool(""),
                "top_sales_invoice": top_sales_tool(""),
                "avg_sales": df['Sales'].mean(),
                "missing_values": df.isnull().sum().to_dict(),
                "correlation_matrix": df.select_dtypes(include='number').corr().to_dict(),
                "tools_used": ["summary_tool","missing_tool","correlation_tool","top_sales_tool","plots_tool"]
            })
        }
    elif choice in ["7","8"]:
        if not user_input:
            return {"output": "No input provided"}
        func = query_tool if choice=="7" else general_question_tool
        return {"output": func(user_input)}
    else:
        tool_map = {"1":"summary_tool","2":"missing_tool","3":"correlation_tool","4":"top_sales_tool","5":"plots_tool"}
        func_name = tool_map[choice]
        func = next(t.func for t in tools if t.name==func_name)
        return {"output": func("")}



