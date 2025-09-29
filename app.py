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

# --- Define table name ---
TABLE_NAME = "FLIGHT_PRICES_RAW"
FULLY_QUALIFIED_TABLE = TABLE_NAME  # Adjust if using schema.db.table

# --- Translate NL prompt to SQL with fallback ---
def translate_to_sql(prompt):
    system_prompt = (
        "You are a Snowflake SQL expert. "
        f"Translate the user's question into a valid SQL query using only the table `{FULLY_QUALIFIED_TABLE}`.\n\n"
        "The table contains flight pricing and segment-level information with the following columns:\n"
        "LEGID, SEARCHDATE, FLIGHTDATE, STARTINGAIRPORT, DESTINATIONAIRPORT, FAREBASISCODE, "
        "TRAVELDURATION, ELAPSEDDAYS, ISBASICECONOMY, ISREFUNDABLE, ISNONSTOP, BASEFARE, TOTALFARE, "
        "SEATSREMAINING, TOTALTRAVELDISTANCE, SEGMENTSDEPARTURETIMEEPOCHSECONDS, "
        "SEGMENTSDEPARTURETIMERAW, SEGMENTSARRIVALTIMEEPOCHSECONDS, SEGMENTSARRIVALTIMERAW, "
        "SEGMENTSARRIVALAIRPORTCODE, SEGMENTSDEPARTUREAIRPORTCODE, SEGMENTSAIRLINENAME, "
        "SEGMENTSAIRLINECODE, SEGMENTSEQUIPMENTDESCRIPTION, SEGMENTSDURATIONINSECONDS, "
        "SEGMENTSDISTANCE, SEGMENTSCABINCODE, INGESTED_AT.\n\n"
        "Notes:\n"
        "- SEGMENTS* fields may contain multiple values separated by '||' (multi-leg flights).\n"
        "- ELAPSEDDAYS = days between SEARCHDATE and FLIGHTDATE.\n"
        "- BASEFARE and TOTALFARE are numeric.\n"
        "- ISREFUNDABLE, ISNONSTOP, ISBASICECONOMY are booleans.\n"
        "- FLIGHTDATE is a date.\n\n"
        "Guidelines:\n"
        "1. Use only columns from this table.\n"
        "2. If you use aggregates in ORDER BY (e.g. COUNT, AVG), include them in SELECT.\n"
        "3. Do not include explanations, markdown, or formatting — return valid SQL only."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        model_used = "gpt-3.5-turbo"
    except Exception as e:
        st.warning(f"gpt-3.5-turbo failed, switching to gpt-4o: {e}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        model_used = "gpt-4o"

    sql = response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = sql.replace("FLIGHTS", FULLY_QUALIFIED_TABLE).replace("Flights", FULLY_QUALIFIED_TABLE)

    st.caption(f"🧠 Generated using: `{model_used}`")

    return sql

# --- Run SQL Query ---
# ...existing code...

# --- Run SQL Query ---
def run_query(sql):
    global conn  # Move global declaration to the top of the function
    try:
        df = pd.read_sql(sql, conn)
        return df
    except snowflake.connector.errors.ProgrammingError as e:
        if "Authentication token has expired" in str(e):
            st.cache_resource.clear()  # Clear cached connection
            st.warning("Session expired. Reconnecting to Snowflake...")
            conn = connect_to_snowflake()
            try:
                df = pd.read_sql(sql, conn)
                return df
            except Exception as e2:
                st.error(f"Error running query after reconnect:\n\n{e2}")
                return None
        else:
            st.error(f"Error running query:\n\n{e}")
            return None
    except Exception as e:
        st.error(f"Error running query:\n\n{e}")
        return None

# ...existing code...

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

        # Optional: bar chart if grouped result with count/total
        if "flight_count" in result_df.columns:
            st.bar_chart(result_df.set_index(result_df.columns[0])["flight_count"])
    else:
        st.warning("No results found or there was an error in the SQL.")