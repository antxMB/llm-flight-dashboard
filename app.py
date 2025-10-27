import streamlit as st
import os
import pandas as pd
import snowflake.connector
import snowflake.connector.errors
import openai
from openai import OpenAI
import plotly.express as px
from dotenv import load_dotenv  # Import dotenv

# --- Load environment variables ---
load_dotenv()  # This will load variables from the .env file into the environment

# --- Page Configuration ---
st.set_page_config(page_title="LLM Flight Dashboard", layout="wide")
st.title("Flight Data Chat Dashboard")

# --- OpenAI client setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Connect to Snowflake ---
@st.cache_resource
def connect_to_snowflake():
    """Connect to Snowflake using Streamlit secrets."""
    try:
        # Check for required configurations
        required_configs = [
            "SNOWFLAKE_USER",
            "SNOWFLAKE_ACCOUNT", 
            "SNOWFLAKE_WAREHOUSE",
            "SNOWFLAKE_DATABASE",
            "SNOWFLAKE_SCHEMA",
            "SNOWFLAKE_ROLE"
        ]
        
        # Verify all required configs exist
        missing_configs = [config for config in required_configs if config not in st.secrets]
        if missing_configs:
            st.error(f"Missing required configurations: {', '.join(missing_configs)}")
            return None

        # Try key-pair authentication first
        if "PRIVATE_KEY" in st.secrets:
            try:
                from cryptography.hazmat.primitives import serialization
                
                # Load and decode the private key
                private_key_str = st.secrets["PRIVATE_KEY"]
                
                # Clean up the private key string - remove any trailing characters
                private_key_str = private_key_str.rstrip('%').strip()
                
                # Ensure proper newline formatting
                if "\\n" in private_key_str:
                    private_key_str = private_key_str.replace("\\n", "\n")
                
                # Load the private key
                private_key = serialization.load_pem_private_key(
                    private_key_str.encode(), 
                    password=None
                )
                
                # Connect using key pair auth
                return snowflake.connector.connect(
                    user=st.secrets["SNOWFLAKE_USER"],
                    account=st.secrets["SNOWFLAKE_ACCOUNT"],
                    private_key=private_key,
                    warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
                    database=st.secrets["SNOWFLAKE_DATABASE"],
                    schema=st.secrets["SNOWFLAKE_SCHEMA"],
                    role=st.secrets["SNOWFLAKE_ROLE"]
                )
                
            except Exception as e:
                st.error(f"Private key authentication failed: {str(e)}")
                return None
        
        # Fall back to password auth if configured
        elif "SNOWFLAKE_PASSWORD" in st.secrets:
            return snowflake.connector.connect(
                user=st.secrets["SNOWFLAKE_USER"],
                password=st.secrets["SNOWFLAKE_PASSWORD"],
                account=st.secrets["SNOWFLAKE_ACCOUNT"],
                warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
                database=st.secrets["SNOWFLAKE_DATABASE"],
                schema=st.secrets["SNOWFLAKE_SCHEMA"],
                role=st.secrets["SNOWFLAKE_ROLE"]
            )
        else:
            st.error("No valid authentication method configured. Please provide either PRIVATE_KEY or SNOWFLAKE_PASSWORD in secrets.")
            return None
            
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {str(e)}")
        return None

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
    global conn
    try:
        df = pd.read_sql(sql, conn)
        df.columns = df.columns.str.lower()  # Normalize column names to lowercase
        return df
    except snowflake.connector.errors.ProgrammingError as e:
        if "Authentication token has expired" in str(e):
            st.cache_resource.clear()  # Clear cached connection
            st.warning("Session expired. Reconnecting to Snowflake...")
            conn = connect_to_snowflake()  # Reconnect
            try:
                df = pd.read_sql(sql, conn)
                df.columns = df.columns.str.lower()  # Normalize column names to lowercase
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

# --- Main App Logic ---
user_query = st.text_input("Enter your query:", "")

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

# --- Tabs for Features ---
tab1, tab2 = st.tabs(["Revenue Insights", "Operational Metrics"])

# --- Revenue Insights ---
with tab1:
    st.header("Revenue Insights")

    # Fare Trends Over Time
    st.subheader("Fare Trends Over Time")
    sql = f"""
        SELECT FLIGHTDATE, AVG(BASEFARE) AS avg_basefare, AVG(TOTALFARE) AS avg_totalfare
        FROM {FULLY_QUALIFIED_TABLE}
        GROUP BY FLIGHTDATE
        ORDER BY FLIGHTDATE
    """
    fare_trends = run_query(sql)

    if fare_trends is not None:
        st.write("Fare Trends DataFrame:")
        st.dataframe(fare_trends)  # Debugging: Display the DataFrame

        # Normalize column names to lowercase
        fare_trends.columns = fare_trends.columns.str.lower()

        # Check if the expected columns exist
        if "avg_basefare" in fare_trends.columns and "avg_totalfare" in fare_trends.columns:
            # Ensure correct data types
            fare_trends["flightdate"] = pd.to_datetime(fare_trends["flightdate"])
            fare_trends["avg_basefare"] = pd.to_numeric(fare_trends["avg_basefare"], errors="coerce")
            fare_trends["avg_totalfare"] = pd.to_numeric(fare_trends["avg_totalfare"], errors="coerce")

            # Handle missing values
            fare_trends = fare_trends.dropna()

            # Plot the line chart
            fig = px.line(
                fare_trends,
                x="flightdate",
                y=["avg_basefare", "avg_totalfare"],
                title="Average Fare Trends Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Expected columns 'avg_basefare' and 'avg_totalfare' are missing from the query results.")
    else:
        st.warning("No data available for Fare Trends Over Time.")

    # Top Revenue Routes
    st.subheader("Top Revenue Routes")
    sql = f"""
        SELECT STARTINGAIRPORT, DESTINATIONAIRPORT, SUM(TOTALFARE) AS total_revenue
        FROM {FULLY_QUALIFIED_TABLE}
        GROUP BY STARTINGAIRPORT, DESTINATIONAIRPORT
        ORDER BY total_revenue DESC
        LIMIT 10
    """
    top_routes = run_query(sql)
    if top_routes is not None:
        st.write("Top Revenue Routes DataFrame:")
        st.dataframe(top_routes)  # Debugging: Display the DataFrame
        st.write("Column names in top_routes DataFrame:", top_routes.columns.tolist())  # Debugging: Display column names

        fig = px.bar(
            top_routes,
            x="total_revenue",  # Use lowercase column names
            y="startingairport",
            color="destinationairport",
            orientation="h",
            title="Top Revenue Routes"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Airline Fare Comparison
    st.subheader("Airline Fare Comparison")
    sql = f"""
        SELECT SEGMENTSAIRLINENAME, AVG(BASEFARE) AS avg_basefare, AVG(TOTALFARE) AS avg_totalfare
        FROM {FULLY_QUALIFIED_TABLE}
        GROUP BY SEGMENTSAIRLINENAME
        ORDER BY avg_totalfare DESC
    """
    airline_fares = run_query(sql)
    if airline_fares is not None:
        st.write("Airline Fare Comparison DataFrame:")
        st.dataframe(airline_fares)  # Debugging: Display the DataFrame
        st.write("Column names in airline_fares DataFrame:", airline_fares.columns.tolist())  # Debugging: Display column names

        fig = px.bar(
            airline_fares,
            x="segmentsairlinename",  # Use lowercase column names
            y=["avg_basefare", "avg_totalfare"],
            barmode="group",
            title="Airline Fare Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Operational Metrics ---
with tab2:
    st.header("Operational Metrics")

    # Flight Load Factor by Route
    st.subheader("Flight Load Factor by Route")
    sql = f"""
        SELECT STARTINGAIRPORT, DESTINATIONAIRPORT, AVG(SEATSREMAINING) AS avg_seats_remaining
        FROM {FULLY_QUALIFIED_TABLE}
        GROUP BY STARTINGAIRPORT, DESTINATIONAIRPORT
        ORDER BY avg_seats_remaining ASC
        LIMIT 10
    """
    load_factors = run_query(sql)
    if load_factors is not None:
        st.write("Column names in load_factors DataFrame:", load_factors.columns.tolist())  # Debugging: Display column names
        st.dataframe(load_factors)  # Debugging: Display the DataFrame

        fig = px.bar(
            load_factors,
            x="avg_seats_remaining",  # Use lowercase column names
            y="startingairport",
            color="destinationairport",
            orientation="h",
            title="Flight Load Factor by Route"
        )
        st.plotly_chart(fig, use_container_width=True)

    # At-Risk Flights (Low Seats)
    st.subheader("At-Risk Flights (Low Seats)")
    sql = f"""
        SELECT FLIGHTDATE, STARTINGAIRPORT, DESTINATIONAIRPORT, SEATSREMAINING
        FROM {FULLY_QUALIFIED_TABLE}
        WHERE SEATSREMAINING < 10
        ORDER BY SEATSREMAINING ASC
        LIMIT 20
    """
    at_risk_flights = run_query(sql)
    if at_risk_flights is not None:
        st.dataframe(at_risk_flights, use_container_width=True)

    # Travel Duration by Route
    st.subheader("Travel Duration by Route")
    sql = f"""
        SELECT 
            STARTINGAIRPORT, 
            DESTINATIONAIRPORT, 
            AVG(
                CASE 
                    WHEN TRAVELDURATION LIKE 'PT%H%M' THEN 
                        TRY_TO_NUMBER(SUBSTR(TRAVELDURATION, 3, POSITION('H' IN TRAVELDURATION) - 3)) * 3600 + 
                        TRY_TO_NUMBER(SUBSTR(TRAVELDURATION, POSITION('H' IN TRAVELDURATION) + 1, POSITION('M' IN TRAVELDURATION) - POSITION('H' IN TRAVELDURATION) - 1)) * 60
                    WHEN TRAVELDURATION LIKE 'PT%H' THEN 
                        TRY_TO_NUMBER(SUBSTR(TRAVELDURATION, 3, POSITION('H' IN TRAVELDURATION) - 3)) * 3600
                    WHEN TRAVELDURATION LIKE 'PT%M' THEN 
                        TRY_TO_NUMBER(SUBSTR(TRAVELDURATION, 3, POSITION('M' IN TRAVELDURATION) - 3)) * 60
                    ELSE NULL
                END
            ) AS avg_travel_duration
        FROM {FULLY_QUALIFIED_TABLE}
        GROUP BY STARTINGAIRPORT, DESTINATIONAIRPORT
        ORDER BY avg_travel_duration DESC
        LIMIT 10
    """
    travel_durations = run_query(sql)
    if travel_durations is not None:
        st.write("Travel Duration DataFrame:")
        st.dataframe(travel_durations)  # Debugging: Display the DataFrame

        fig = px.bar(
            travel_durations,
            x="avg_travel_duration",
            y="startingairport",
            color="destinationairport",
            orientation="h",
            title="Travel Duration by Route"
        )
        st.plotly_chart(fig, use_container_width=True)