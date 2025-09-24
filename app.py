import streamlit as st
import os
import pandas as pd
import snowflake.connector
import openai

# --- Page Configuration ---
st.set_page_config(page_title="LLM Flight Dashboard", layout="wide")
st.title("✈️ Flight Data Chat Dashboard")

# --- OpenAI client setup (v1+ syntax) ---
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- User input for NL question ---
user_query = st.text_input("Ask your question about flight data:")

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

# --- Translate NL prompt to SQL ---
def translate_to_sql(prompt):
    system_prompt = (
        "You are an expert SQL assistant. "
        "Translate the user's natural language question into a valid Snowflake SQL query. "
        "ONLY return the SQL query without any Markdown formatting (no ```sql or explanations)."
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    sql = response.choices[0].message.content.strip()

    # 🔥 Remove any triple backticks if the model adds them anyway
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql

# --- Execute the SQL query ---
def run_query(sql):
    try:
        df = pd.read_sql(sql, conn)
        return df
    except Exception as e:
        st.error(f"Error running query:\n\n{e}")
        return None

# --- Main Logic ---
if user_query:
    with st.spinner("Generating SQL..."):
        generated_sql = translate_to_sql(user_query)

    st.markdown("### 🧠 Generated SQL")
    st.code(generated_sql, language="sql")

    with st.spinner("Running query..."):
        result_df = run_query(generated_sql)

    if result_df is not None and not result_df.empty:
        st.markdown("### 📊 Query Results")
        st.dataframe(result_df, use_container_width=True)
    else:
        st.warning("No results found or there was an error in the SQL.")