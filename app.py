import streamlit as st
import os
import pandas as pd
import snowflake.connector
import openai

# --- UI: Page Config ---
st.set_page_config(page_title="LLM Flight Dashboard", layout="wide")
st.title("Flight Data Chat Dashboard")

# --- Prompt input from user ---
user_query = st.text_input("Ask your question about flight data:")

# --- OpenAI client setup (v1+ syntax) ---
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Snowflake connection using Streamlit secrets ---
@st.cache_resource
def connect_to_snowflake():
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )

conn = connect_to_snowflake()

# --- Function: Translate user query into SQL using OpenAI ---
def translate_to_sql(prompt):
    system_prompt = (
        "You are an expert SQL generator for Snowflake. "
        "Translate natural language questions into valid, optimized SQL queries "
        "based on a flight pricing dataset. Only return the SQL query."
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# --- Function: Run SQL and return DataFrame ---
def run_query(sql):
    try:
        df = pd.read_sql(sql, conn)
        return df
    except Exception as e:
        st.error(f"Error running query: {e}")
        return None

# --- Main app logic ---
if user_query:
    st.markdown("### Generated SQL")
    generated_sql = translate_to_sql(user_query)
    st.code(generated_sql, language="sql")

    st.markdown("### Query Results")
    result_df = run_query(generated_sql)

    if result_df is not None and not result_df.empty:
        st.dataframe(result_df)
    else:
        st.warning("No results returned or an error occurred.")