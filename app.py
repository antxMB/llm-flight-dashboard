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

# --- Get user input ---
user_query = st.text_input("Ask your question about flight data:")

# --- Connect to Snowflake ---
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

# --- Define the schema (column names) ---
TABLE_NAME = "FLIGHT_PRICES_RAW"  # ✅ your confirmed table name
FULLY_QUALIFIED_TABLE = TABLE_NAME  # update this if needed to e.g. MY_DB.PUBLIC.FLIGHT_PRICES_RAW

TABLE_COLUMNS = (
    "LEGID, SEARCHDATE, FLIGHTDATE, STARTINGAIRPORT, DESTINATIONAIRPORT, FAREBASISCODE, "
    "TRAVELDURATION, ELAPSEDDAYS, ISBASICECONOMY, ISREFUNDABLE, ISNONSTOP, BASEFARE, TOTALFARE, "
    "SEATSREMAINING, TOTALTRAVELDISTANCE, SEGMENTSDEPARTURETIMEEPOCHSECONDS, "
    "SEGMENTSDEPARTURETIMERAW, SEGMENTSARRIVALTIMEEPOCHSECONDS, SEGMENTSARRIVALTIMERAW, "
    "SEGMENTSARRIVALAIRPORTCODE, SEGMENTSDEPARTUREAIRPORTCODE, SEGMENTSAIRLINENAME, "
    "SEGMENTSAIRLINECODE, SEGMENTSEQUIPMENTDESCRIPTION, SEGMENTSDURATIONINSECONDS, "
    "SEGMENTSDISTANCE, SEGMENTSCABINCODE, INGESTED_AT"
)

# --- Translate natural language to SQL ---
def translate_to_sql(prompt):
    system_prompt = (
        f"You are an expert SQL assistant working with Snowflake. "
        f"The table is `{FULLY_QUALIFIED_TABLE}` and contains the following columns:\n{TABLE_COLUMNS}.\n"
        f"Generate a valid SQL query based only on this table. "
        f"Only return the SQL query with NO explanation and NO markdown (no triple backticks)."
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    sql = response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    # Optional: enforce correct table name if LLM gets creative
    sql = sql.replace("FLIGHTS", FULLY_QUALIFIED_TABLE).replace("Flights", FULLY_QUALIFIED_TABLE)

    return sql

# --- Run SQL Query ---
def run_query(sql):
    try:
        df = pd.read_sql(sql, conn)
        return df
    except Exception as e:
        st.error(f"Error running query:\n\n{e}")
        return None

# --- Main App Logic ---
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