import streamlit as st
import openai
import os
from dotenv import load_dotenv
import pandas as pd
import snowflake.connector

load_dotenv()

# Secrets from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.title("✈️ Flight Pricing LLM Dashboard")
user_role = st.selectbox("Select your persona", ["Revenue Manager", "Operations Controller"])
user_query = st.text_input("Ask your question about flight data:")

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)

def query_snowflake(sql):
    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall(), [desc[0] for desc in cur.description]

if user_query:
    with st.spinner("Thinking..."):
        prompt = f"You are a helpful SQL assistant for aviation data. Role: {user_role}. Write a Snowflake SQL query based on this: {user_query}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        sql_query = response['choices'][0]['message']['content']
        st.code(sql_query, language='sql')
        
        try:
            rows, cols = query_snowflake(sql_query)
            df = pd.DataFrame(rows, columns=cols)
            st.dataframe(df)
        except Exception as e:
            st.error(f"SQL Error: {e}")