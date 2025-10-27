import streamlit as st
st.set_page_config(page_title="LLM Flight Dashboard", layout="wide")

# Rest of your imports
import os
import pandas as pd
import snowflake.connector
import openai
import snowflake.connector.errors
import plotly.express as px
from openai import OpenAI
from datetime import date, timedelta
from cryptography.hazmat.primitives import serialization  # Add this import

# --- Define table name ---
TABLE_NAME = "FLIGHT_PRICES_RAW"
FULLY_QUALIFIED_TABLE = TABLE_NAME

# --- Run SQL Query ---
def run_query(sql):
    def execute_query(conn):
        try:
            df = pd.read_sql(sql, conn)
            df.columns = df.columns.str.lower()
            return df
        except Exception as e:
            error_msg = str(e)
            if "Authentication token has expired" in error_msg:
                return None
            elif "Division by zero" in error_msg:
                st.error("⚠️ Query failed: Cannot divide by zero. Please modify your query.")
            elif "syntax error" in error_msg.lower():
                st.error("⚠️ Query failed: Invalid SQL syntax.")
            else:
                st.error("⚠️ Query failed. Please try a different query.")
            return None

    # First attempt
    if st.session_state.snowflake_conn is not None:
        result = execute_query(st.session_state.snowflake_conn)
        if result is not None:
            return result

    # If first attempt failed, try reconnecting
    st.session_state.snowflake_conn = connect_to_snowflake()
    if st.session_state.snowflake_conn is not None:
        result = execute_query(st.session_state.snowflake_conn)
        if result is not None:
            return result
    
    return None

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
                # Load and decode the private key - handle different newline formats
                private_key_str = st.secrets["PRIVATE_KEY"]
                
                # Clean up the private key string
                # Remove any trailing characters like '%'
                private_key_str = private_key_str.rstrip('%').strip()
                
                # Ensure proper newline formatting
                if "\\n" in private_key_str:
                    private_key_str = private_key_str.replace("\\n", "\n")
                
                # Validate the key format
                if not private_key_str.startswith("-----BEGIN PRIVATE KEY-----"):
                    st.error("❌ Private key doesn't start with -----BEGIN PRIVATE KEY-----")
                    return None
                
                if not private_key_str.endswith("-----END PRIVATE KEY-----"):
                    st.error("❌ Private key doesn't end with -----END PRIVATE KEY-----")
                    return None
                
                # Load the private key
                try:
                    private_key = serialization.load_pem_private_key(
                        private_key_str.encode(), 
                        password=None
                    )
                except Exception as parse_error:
                    st.error(f"❌ Failed to parse private key: {parse_error}")
                    return None
                
                # Connect using key pair auth
                conn = snowflake.connector.connect(
                    user=st.secrets["SNOWFLAKE_USER"],
                    account=st.secrets["SNOWFLAKE_ACCOUNT"],
                    private_key=private_key,
                    warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
                    database=st.secrets["SNOWFLAKE_DATABASE"],
                    schema=st.secrets["SNOWFLAKE_SCHEMA"],
                    role=st.secrets["SNOWFLAKE_ROLE"]
                )
                return conn
            except snowflake.connector.errors.DatabaseError as db_error:
                st.error(f"❌ Snowflake database error: {str(db_error)}")
                return None
            except Exception as e:
                st.error(f"❌ Private key authentication failed: {str(e)}")
                st.error("Please check your private key format in the secrets configuration")
                return None
        
        # Only try password auth if explicitly configured
        elif "SNOWFLAKE_PASSWORD" in st.secrets:
            st.write("🔗 Attempting password authentication...")
            conn = snowflake.connector.connect(
                user=st.secrets["SNOWFLAKE_USER"],
                password=st.secrets["SNOWFLAKE_PASSWORD"],
                account=st.secrets["SNOWFLAKE_ACCOUNT"],
                warehouse=st.secrets["SNOWFLAKE_WAREHOUSE"],
                database=st.secrets["SNOWFLAKE_DATABASE"],
                schema=st.secrets["SNOWFLAKE_SCHEMA"],
                role=st.secrets["SNOWFLAKE_ROLE"]
            )
            return conn
        else:
            st.error("❌ No valid authentication method configured. Please provide either PRIVATE_KEY or SNOWFLAKE_PASSWORD in secrets.")
            return None
            
    except Exception as e:
        st.error("❌ Failed to connect to Snowflake")
        st.error(f"Error details: {str(e)}")
        return None

# --- Add connection state management ---
if 'snowflake_conn' not in st.session_state:
    st.session_state.snowflake_conn = connect_to_snowflake()

# Title and description
st.title("Flight Data Chat Dashboard")
st.markdown("""
### 📋 Dataset Overview
Flight pricing information from Expedia (2022-04-16 to 2022-10-05)

**Each row represents:**
- A purchasable flight ticket found on a specific search date
- Including details about price, route, and availability
""")


# Add persona selection at the top of sidebar
with st.sidebar:
    persona = st.radio(
        "Select your role:",
        ["Revenue Manager (Strategic)", "Ops Controller (Tactical)"],
        help="Choose your role for this analysis session. It will shape how insights are presented."
    )
    
    st.markdown("---")  # Add separator after role selection

    # --- Date Range Filter ---
    flight_start_date = st.sidebar.date_input(
        "Select Start Date:",
        value=date(2022, 4, 16),
        min_value=date(2022, 4, 16),
        max_value=date(2022, 10, 5)
    )
    
    flight_end_date = st.sidebar.date_input(
        "Select End Date:",
        value=date(2022, 10, 5),  # Always default to dataset end
        min_value=flight_start_date,  # Can't be before start date
        max_value=date(2022, 10, 5)
    )
    
    # --- Airport Filters ---
    starting_airports_query = run_query(f"SELECT DISTINCT STARTINGAIRPORT FROM {FULLY_QUALIFIED_TABLE}")
    starting_airport_options = (["All"] + starting_airports_query.startingairport.tolist()) if starting_airports_query is not None else ["All"]
    starting_airport = st.selectbox(
        "Select Starting Airport:",
        options=starting_airport_options,
        index=0
    )

    destination_airports_query = run_query(f"SELECT DISTINCT DESTINATIONAIRPORT FROM {FULLY_QUALIFIED_TABLE}")
    destination_airport_options = (["All"] + destination_airports_query.destinationairport.tolist()) if destination_airports_query is not None else ["All"]
    destination_airport = st.selectbox(
        "Select Destination Airport:",
        options=destination_airport_options,
        index=0
    )

    # --- Fare Range Filter ---
    max_fare_query = run_query(f"SELECT MAX(TOTALFARE) AS max_fare FROM {FULLY_QUALIFIED_TABLE}")
    default_max_fare = 8260  # Set a reasonable default maximum fare
    max_fare = int(max_fare_query.max_fare[0]) if max_fare_query is not None else default_max_fare

    fare_range = st.slider(
        "Select Fare Range:",
        min_value=0,
        max_value=max_fare,
        value=(0, 8260),
        help="Set the minimum and maximum fare range for filtering"
    )

    # --- Add Reset Filters Button ---
    st.sidebar.markdown("---")  # Add separator
    if st.sidebar.button("🔄 Reset All Filters", key="reset_filters"):  # Added unique key
        # Reset date inputs to default values
        st.session_state['start_date'] = date(2022, 4, 16)
        st.session_state['end_date'] = date(2022, 10, 5)
        # Reset airport selections
        st.session_state['starting_airport'] = 0  # Reset to 'All'
        st.session_state['destination_airport'] = 0  # Reset to 'All'
        # Reset fare range
        st.session_state['fare_range'] = (0, 500)
        # Force a rerun to apply changes
        st.rerun()

    st.markdown("---")  # Add separator before expandable sections

    # Add dataset details in expandable sections
    with st.expander("🔑 Key Metrics", expanded=True):
        st.markdown("""
        - `TOTALFARE`: Total price with taxes/fees
        - `BASEFARE`: Base ticket price
        - `SEATSREMAINING`: Available seats
        - `TOTALTRAVELDISTANCE`: Miles
        - `TRAVELDURATION`: Journey time
        """)
    
    with st.expander("✈️ Route Information", expanded=True):
        st.markdown("""
        - Origin/destination airports
        - Non-stop vs. multi-leg flights
        - Airline details
        - Equipment information
        """)
    
    with st.expander("⏱️ Time Dimensions", expanded=True):
        st.markdown("""
        - `SEARCHDATE`: Fare found date
        - `FLIGHTDATE`: Departure date
        - `ELAPSEDDAYS`: Search to flight
        """)

# --- OpenAI client setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # Load your key securely

# --- Translate NL prompt to SQL with fallback ---
def translate_to_sql(prompt):
    system_prompt = (
    "You are a Snowflake SQL expert supporting an Operations Controller. "
    f"Translate the user's operational question into a valid SQL query using only the table `{FULLY_QUALIFIED_TABLE}`.\n\n"
    "This table contains flight pricing and segment-level information with the following columns:\n"
    "LEGID, SEARCHDATE, FLIGHTDATE, STARTINGAIRPORT, DESTINATIONAIRPORT, FAREBASISCODE, "
    "TRAVELDURATION (ISO 8601 format e.g., 'PT13H19M'), ELAPSEDDAYS, ISBASICECONOMY, ISREFUNDABLE, ISNONSTOP, "
    "BASEFARE, TOTALFARE, SEATSREMAINING, TOTALTRAVELDISTANCE, "
    "SEGMENTSDEPARTURETIMEEPOCHSECONDS, SEGMENTSDEPARTURETIMERAW, SEGMENTSARRIVALTIMEEPOCHSECONDS, "
    "SEGMENTSARRIVALTIMERAW, SEGMENTSARRIVALAIRPORTCODE, SEGMENTSDEPARTUREAIRPORTCODE, "
    "SEGMENTSAIRLINENAME, SEGMENTSAIRLINECODE, SEGMENTSEQUIPMENTDESCRIPTION, "
    "SEGMENTSDURATIONINSECONDS, SEGMENTSDISTANCE, SEGMENTSCABINCODE, INGESTED_AT.\n\n"
    "Notes:\n"
    "- TRAVELDURATION is stored as ISO 8601 (e.g., 'PT13H19M'). Use REGEXP_SUBSTR to extract hours.\n"
    "- SEGMENTS* fields may contain multiple values separated by '||' (multi-leg flights).\n"
    "- ELAPSEDDAYS = days between SEARCHDATE and FLIGHTDATE.\n"
    "- BASEFARE and TOTALFARE are numeric.\n"
    "- ISREFUNDABLE, ISNONSTOP, ISBASICECONOMY are boolean.\n"
    "- FLIGHTDATE is a DATE type.\n\n"
    "Guidelines:\n"
    "1. Use only columns from this table.\n"
    "2. Use REGEXP_SUBSTR(TRAVELDURATION, 'PT(\\d+)H')::int for duration comparisons.\n"
    "3. Always include aggregate expressions used in ORDER BY inside SELECT.\n"
    "4. Return ONLY the SQL query with no explanations, formatting, or comments.\n"
    "5. The SQL must start with SELECT and end with a semicolon.\n"
    "6. Do not reference or JOIN other tables.\n"
    "7. Compare boolean columns using `= TRUE` or `= FALSE`.\n"
    "8. Ignore NULL values unless explicitly mentioned.\n"
    "9. Avoid interpreting text like 'PT45M' unless asked — default to 'PT(\d+)H'.\n"
    "10. Focus on operational aspects like route load, availability, timing, and delays.\n"
    "11. Use FLIGHTDATE or SEARCHDATE for time-based filtering when prompted.\n"
    "12. Use aliases (`AS`) for computed columns when useful for readability."
)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        resp = client.chat.completions.create(model="gpt-4", messages=messages)
        model_used = "gpt-4"
    except Exception as e:
        st.warning(f"gpt-4 failed, switching to gpt-3.5-turbo: {e}")
        try:
            resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
            model_used = "gpt-3.5-turbo"
        except Exception as e2:
            st.error("❌ Failed to generate SQL query")
            return None

    try:
        content = resp.choices[0].message.content
        # Extract only the SQL query
        sql = content.strip()
        
        # Remove any markdown formatting
        sql = sql.replace("```sql", "").replace("```", "")
        
        # Remove any explanatory text before or after the query
        if "SELECT" in sql.upper():
            sql = sql[sql.upper().find("SELECT"):]
            if ";" in sql:
                sql = sql[:sql.find(";") + 1]
        
        # Validate it's a proper SQL query
        if not sql.upper().startswith("SELECT"):
            st.error("❌ Invalid SQL generated. Please rephrase your question.")
            return None

        return sql

    except Exception as e:
        st.error("❌ Error processing AI response")
        return None

# --- Insights Generator ---
def insights_generator(df):
    """Generate concise operational insights from query results."""
    if df is None or df.empty:
        st.info("No results to generate insights.")
        return

    rows = len(df)
    cols = set(df.columns.str.lower())
    lines = [f"**Analysis based on {rows} records.**\n"]

    # Seats / capacity insights
    if 'seatsremaining' in cols:
        s = df['seatsremaining'].dropna().astype(float)
        if not s.empty:
            avg = s.mean()
            med = s.median()
            mn = s.min()
            mx = s.max()
            pct_critical = (s < 5).mean() * 100
            lines.append("**Capacity summary:**")
            lines.append(f"- Average seats available: {avg:.1f}")
            lines.append(f"- Median: {med:.1f}, Min: {mn:.0f}, Max: {mx:.0f}")
            lines.append(f"- Routes / records with <5 seats: {pct_critical:.1f}%\n")

            # top routes by avg seats if route columns exist
            if 'startingairport' in cols and 'destinationairport' in cols:
                route_avg = (
                    df.groupby(['startingairport', 'destinationairport'], as_index=False)
                      .seatsremaining.mean()
                      .rename(columns={'seatsremaining': 'avg_seats'})
                )
                top_low = route_avg.nsmallest(3, 'avg_seats')
                top_high = route_avg.nlargest(3, 'avg_seats')
                if not top_low.empty:
                    lines.append("**Most constrained routes (lowest avg seats):**")
                    for _, r in top_low.iterrows():
                        lines.append(f"- {r['startingairport']} → {r['destinationairport']}: {r['avg_seats']:.1f} seats")
                    lines.append("")
                if not top_high.empty:
                    lines.append("**Most underutilized routes (highest avg seats):**")
                    for _, r in top_high.iterrows():
                        lines.append(f"- {r['startingairport']} → {r['destinationairport']}: {r['avg_seats']:.1f} seats")
                    lines.append("")

    # Distance / short-haul insights
    if 'totaltraveldistance' in cols:
        d = pd.to_numeric(df['totaltraveldistance'], errors='coerce').dropna()
        if not d.empty:
            avg_d = d.mean()
            pct_short = (d <= 500).mean() * 100
            lines.append("**Distance summary:**")
            lines.append(f"- Average route distance: {avg_d:.0f} miles")
            lines.append(f"- Short-haul (≤500 miles) in results: {pct_short:.1f}%\n")

    # Route coverage
    if 'startingairport' in cols or 'destinationairport' in cols:
        origins = int(df['startingairport'].nunique()) if 'startingairport' in cols else 0
        dests = int(df['destinationairport'].nunique()) if 'destinationairport' in cols else 0
        lines.append("**Route coverage:**")
        lines.append(f"- Unique origins in result: {origins}")
        lines.append(f"- Unique destinations in result: {dests}\n")

    # If fare exists, give quick stats (ops may find useful)
    if 'totalfare' in cols:
        f = pd.to_numeric(df['totalfare'], errors='coerce').dropna()
        if not f.empty:
            lines.append("**Fare snapshot:**")
            lines.append(f"- Avg fare: ${f.mean():.0f}, Median: ${f.median():.0f}, Max: ${f.max():.0f}\n")

    # If nothing beyond record count was produced, provide guidance
    if len(lines) == 1:
        lines.append("_No operational columns (seats, distance or route) found in the query result. Try a query that returns seatsremaining, startingairport and destinationairport for richer insights._")

    # Render single markdown block
    st.markdown("\n".join(lines))

# --- Main App Logic ---
# --- Construct WHERE clause ---
where_conditions = [
    f"FLIGHTDATE BETWEEN '{flight_start_date}' AND '{flight_end_date}'",
    f"TOTALFARE BETWEEN {fare_range[0]} AND {fare_range[1]}"
]

if starting_airport != "All":
    where_conditions.append(f"STARTINGAIRPORT = '{starting_airport}'")
if destination_airport != "All":
    where_conditions.append(f"DESTINATIONAIRPORT = '{destination_airport}'")

where_clause = " AND ".join(where_conditions)

# --- Content Structure ---
st.markdown("---")

if persona == "Revenue Manager (Strategic)":
    # --- Revenue Manager Content ---
    st.markdown("---")
    # Fare Trends Over Time
    st.subheader("Fare Trends Over Time")
    sql = f"""
        SELECT FLIGHTDATE, AVG(BASEFARE) AS avg_basefare, AVG(TOTALFARE) AS avg_totalfare
        FROM {FULLY_QUALIFIED_TABLE}
        WHERE {where_clause}
        GROUP BY FLIGHTDATE
        ORDER BY FLIGHTDATE
    """
    fare_trends = run_query(sql)

    if fare_trends is not None:
        fare_trends["avg_basefare"] = fare_trends["avg_basefare"].round(2)
        fare_trends["avg_totalfare"] = fare_trends["avg_totalfare"].round(2)

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

    st.markdown("---")  # Add separator before next major section
    # Top Revenue Routes
    st.subheader("Top Revenue Routes")
    sql = f"""
        SELECT STARTINGAIRPORT, DESTINATIONAIRPORT, SUM(TOTALFARE) AS total_revenue
        FROM {FULLY_QUALIFIED_TABLE}
        WHERE {where_clause}
        GROUP BY STARTINGAIRPORT, DESTINATIONAIRPORT
        ORDER BY total_revenue DESC
        LIMIT 10
    """
    top_routes = run_query(sql)
    if top_routes is not None:
        
        fig = px.bar(
            top_routes,
            x="total_revenue",  # Use lowercase column names
            y="startingairport",
            color="destinationairport",
            orientation="h",
            title="Top Revenue Routes"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")  # Add separator
    # Airline Fare Comparison
    st.subheader("Airline Fare Comparison")
    sql = f"""
        SELECT SEGMENTSAIRLINENAME, AVG(BASEFARE) AS avg_basefare, AVG(TOTALFARE) AS avg_totalfare
        FROM {FULLY_QUALIFIED_TABLE}
        WHERE {where_clause}
        GROUP BY SEGMENTSAIRLINENAME
        ORDER BY avg_totalfare DESC
    """
    airline_fares = run_query(sql)
    if airline_fares is not None:
        airline_fares["avg_basefare"] = airline_fares["avg_basefare"].round(2)
        airline_fares["avg_totalfare"] = airline_fares["avg_totalfare"].round(2)

        fig = px.bar(
            airline_fares,
            x="segmentsairlinename",  # Use lowercase column names
            y=["avg_basefare", "avg_totalfare"],
            barmode="group",
            title="Airline Fare Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")  # Add separator
    # Pricing Anomaly Analysis for Short-haul Flights
    st.subheader("Short-haul Flight Pricing Anomalies")
            
    anomaly_query = f"""
    WITH ShortHaulFlights AS (
        SELECT 
            FLIGHTDATE,
            STARTINGAIRPORT,
            DESTINATIONAIRPORT,
            TOTALTRAVELDISTANCE,
            TOTALFARE,
            AVG(TOTALFARE) OVER (
                PARTITION BY STARTINGAIRPORT, DESTINATIONAIRPORT
            ) as avg_route_fare,
            STDDEV(TOTALFARE) OVER (
                PARTITION BY STARTINGAIRPORT, DESTINATIONAIRPORT
            ) as stddev_route_fare
        FROM {FULLY_QUALIFIED_TABLE}
        WHERE 
            TOTALTRAVELDISTANCE <= 500  -- Define short-haul as 500 miles or less
            AND FLIGHTDATE BETWEEN DATEADD('day', -7, '{flight_end_date}') AND '{flight_end_date}'
            AND {where_clause}
    )
    SELECT 
        FLIGHTDATE,
        STARTINGAIRPORT,
        DESTINATIONAIRPORT,
        TOTALTRAVELDISTANCE as distance_miles,
        TOTALFARE as fare,
        avg_route_fare,
        stddev_route_fare,
        (TOTALFARE - avg_route_fare) / NULLIF(stddev_route_fare, 0) as z_score,
        ROUND((TOTALFARE / avg_route_fare - 1) * 100, 1) as percent_deviation
    FROM ShortHaulFlights
    WHERE ABS((TOTALFARE - avg_route_fare) / NULLIF(stddev_route_fare, 0)) > 2  -- Show prices > 2 standard deviations from mean
    ORDER BY ABS((TOTALFARE - avg_route_fare) / NULLIF(stddev_route_fare, 0)) DESC
    LIMIT 20
    """
            
    anomalies_df = run_query(anomaly_query)
            
    if anomalies_df is not None and not anomalies_df.empty:
        # Add better explanatory metrics
        anomalies_df['percent_above_avg'] = ((anomalies_df['fare'] - anomalies_df['avg_route_fare']) / anomalies_df['avg_route_fare'] * 100).round(1)
        
        # Recategorize based on percentage deviation instead of z-score
        anomalies_df['price_category'] = pd.cut(
            anomalies_df['percent_above_avg'],
            bins=[-float('inf'), 25, 50, float('inf')],
            labels=['25% above average', '25-50% above average', '50%+ above average']
        )
        
        # Display summary explanation
        st.markdown("""
        ### Understanding the Pricing Anomalies
        
        **Key Metrics Explained:**
        - **Z-score**: Number of standard deviations from the route's average price
        - **Percent Deviation**: How much higher the fare is compared to route average
        - **Distance**: Route distance in miles (short-haul ≤ 500 miles)
        """)
        
        # Show summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Average Price Increase",
                f"{anomalies_df['percent_above_avg'].mean():.1f}%",
                help="Average percentage above normal route prices"
            )
        with col2:
            st.metric(
                "Highest Price Increase",
                f"{anomalies_df['percent_above_avg'].max():.1f}%",
                help="Maximum percentage above normal route prices"
            )
        with col3:
            st.metric(
                "Routes Affected",
                len(anomalies_df['startingairport'].unique()),
                help="Number of unique routes with price anomalies"
            )
        
        # Add scatter plot visualization
        fig = px.scatter(
            anomalies_df,
            x='distance_miles',
            y='fare',
            size='z_score',  # Size of point indicates severity of anomaly
            color='percent_above_avg',  # Color indicates percentage above average
            hover_data=[
                'startingairport',
                'destinationairport',
                'flightdate',
                'percent_above_avg',
                'avg_route_fare'
            ],
            title="Short-haul Flight Pricing Anomalies",
            labels={
                'distance_miles': 'Flight Distance (miles)',
                'fare': 'Current Fare ($)',
                'percent_above_avg': '% Above Average',
                'z_score': 'Anomaly Severity'
            }
        )
        
        # Add reference line for average fares
        fig.add_scatter(
            x=anomalies_df['distance_miles'],
            y=anomalies_df['avg_route_fare'],
            mode='lines',
            name='Average Route Fare',
            line=dict(color='green', dash='dash')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add styled DataFrame
        st.markdown("### Detailed Anomaly Analysis")
        styled_df = anomalies_df[['startingairport', 'destinationairport', 
                                 'fare', 'avg_route_fare', 'percent_above_avg', 
                                 'distance_miles', 'z_score']].style.background_gradient(
            subset=['percent_above_avg', 'z_score'],
            cmap='RdYlGn_r'
        ).format({
            'fare': '{:.2f}',
            'avg_route_fare': '{:.2f}',
            'percent_above_avg': '{:.2f}',
            'distance_miles': '{:.2f}',
            'z_score': '{:.2f}'
        })
        st.dataframe(styled_df, use_container_width=True)

        # Show example routes
        st.markdown("### Example Routes with Highest Price Deviations")
        top_5_routes = anomalies_df.nlargest(5, 'percent_above_avg')
        for _, route in top_5_routes.iterrows():
            st.markdown(f"""
            **{route['startingairport']} → {route['destinationairport']}**
            - Current fare: ${route['fare']:.0f}
            - Typical fare: ${route['avg_route_fare']:.0f}
            - Increase: {route['percent_above_avg']}%
            - Distance: {route['distance_miles']:.0f} miles
            """)
    else:
        st.info("No significant pricing anomalies found in short-haul flights over the last 7 days.")

    st.markdown("---")  # Add separator
    # Pricing Margin Analysis
    st.subheader("Route Pricing Margin Analysis")
            
    margin_query = f"""
    WITH RouteMetrics AS (
        SELECT 
            FLIGHTDATE,
            STARTINGAIRPORT,
            DESTINATIONAIRPORT,
            AVG(TOTALFARE) as avg_fare,
            AVG(BASEFARE) as avg_base_fare,
            AVG(SEATSREMAINING) as avg_seats_remaining,
            MAX(SEATSREMAINING) as max_seats,
            COUNT(*) as flight_count,
            AVG((TOTALFARE - BASEFARE) / NULLIF(BASEFARE, 0) * 100) as margin_percentage,
            CASE 
                WHEN MAX(SEATSREMAINING) > 0 THEN (1 - (AVG(SEATSREMAINING) / NULLIF(MAX(SEATSREMAINING), 0))) * 100
                ELSE 50  -- Default to 50% occupancy when no seat data
            END as occupancy_rate
        FROM {FULLY_QUALIFIED_TABLE}
        WHERE {where_clause}
        GROUP BY STARTINGAIRPORT, DESTINATIONAIRPORT, FLIGHTDATE
    )
    SELECT 
        STARTINGAIRPORT,
        DESTINATIONAIRPORT,
        AVG(avg_fare) as current_avg_fare,
        AVG(margin_percentage) as margin_percentage,
        AVG(occupancy_rate) as occupancy_percentage,
        SUM(flight_count) as total_flights,
        CASE 
            WHEN AVG(occupancy_rate) >= 70 THEN 'High'
            WHEN AVG(occupancy_rate) >= 50 THEN 'Medium'
            ELSE 'Low'
        END as margin_opportunity
    FROM RouteMetrics
    GROUP BY STARTINGAIRPORT, DESTINATIONAIRPORT
    HAVING total_flights >= 5
    ORDER BY occupancy_percentage DESC
    LIMIT 10
    """
            
    margin_df = run_query(margin_query)
            
    if margin_df is not None and not margin_df.empty:
        # Create visualization
        fig = px.scatter(
            margin_df,
            x='occupancy_percentage',
            y='margin_percentage',
            size='total_flights',
            color='margin_opportunity',
            hover_data=[
                'startingairport',
                'destinationairport', 
                'current_avg_fare'
            ],
            labels={
                'occupancy_percentage': 'Occupancy Rate (%)',
                'margin_percentage': 'Margin (%)',
                'margin_opportunity': 'Opportunity Level'
            },
            title="Route Pricing Margin Opportunities",
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'blue'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display full analysis table
        st.markdown("### Detailed Route Analysis")
        st.dataframe(
            margin_df.style.background_gradient(
                subset=['occupancy_percentage', 'margin_percentage'],
                cmap='RdYlGn'
            ).format({
        'current_avg_fare': '${:.2f}',
        'margin_percentage': '{:.1f}%',
        'price_trend': '${:.2f}',
        'occupancy_percentage': '{:.1f}%'
    }),
    use_container_width=True
    )
    else:
        st.warning("Unable to analyze route margins with current filters.")

    st.markdown("---")  # Add separator
    # Top 5 Destinations Fare Analysis
    st.subheader("Top 5 Destinations Fare Patterns")

    top_destinations_query = f"""
    WITH TopDestinations AS (
        SELECT 
            DESTINATIONAIRPORT,
            COUNT(*) as flight_count,
            AVG(TOTALFARE) as overall_avg_fare
        FROM {FULLY_QUALIFIED_TABLE}
        WHERE {where_clause}
        GROUP BY DESTINATIONAIRPORT
        ORDER BY flight_count DESC
        LIMIT 5
    ),
    DailyMetrics AS (
        SELECT 
            f.DESTINATIONAIRPORT,
            f.FLIGHTDATE,
            AVG(f.TOTALFARE) as daily_avg_fare,
            COUNT(*) as daily_flights,
            MIN(f.TOTALFARE) as min_fare,
            MAX(f.TOTALFARE) as max_fare
        FROM {FULLY_QUALIFIED_TABLE} f
        WHERE 
            f.DESTINATIONAIRPORT IN (SELECT DESTINATIONAIRPORT FROM TopDestinations)
            AND {where_clause}
        GROUP BY f.DESTINATIONAIRPORT, f.FLIGHTDATE
    )
    SELECT 
        dm.FLIGHTDATE,
        dm.DESTINATIONAIRPORT,
        dm.daily_avg_fare,
        dm.daily_flights,
        dm.min_fare,
        dm.max_fare,
        td.overall_avg_fare
    FROM DailyMetrics dm
    JOIN TopDestinations td 
        ON dm.DESTINATIONAIRPORT = td.DESTINATIONAIRPORT
    ORDER BY dm.FLIGHTDATE, dm.DESTINATIONAIRPORT
    """

    top_dest_df = run_query(top_destinations_query)

    if top_dest_df is not None and not top_dest_df.empty:
        for col in ['daily_avg_fare', 'min_fare', 'max_fare']:
            if col in top_dest_df.columns:
                top_dest_df[col] = top_dest_df[col].round(2)
        
        # Create fare trend visualization
        fig = px.line(
            top_dest_df,
            x='flightdate',
            y='daily_avg_fare',
            color='destinationairport',
            title="Average Fare Trends for Top 5 Destinations",
            labels={
                'flightdate': 'Date',
                'daily_avg_fare': 'Average Fare ($)',
                'destinationairport': 'Destination'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics by destination
        st.markdown("### Summary Statistics")
        summary_df = top_dest_df.groupby('destinationairport').agg({
            'daily_avg_fare': ['mean', 'min', 'max', 'std'],
            'daily_flights': 'mean'
        }).round(2)
        
        summary_df.columns = ['Avg Fare', 'Min Fare', 'Max Fare', 'Std Dev', 'Avg Daily Flights']
        st.dataframe(
            summary_df.style.background_gradient(
                subset=['Avg Fare'],
                cmap='YlOrRd'
            ),
            use_container_width=True
        )
        
        # Fare variation analysis
        st.markdown("### Fare Patterns")
        for dest in top_dest_df['destinationairport'].unique():
            dest_data = top_dest_df[top_dest_df['destinationairport'] == dest]
            avg_fare = dest_data['daily_avg_fare'].mean()
            fare_volatility = dest_data['daily_avg_fare'].std() / avg_fare * 100
            
            st.markdown(f"""
            **{dest}**:
            - Average fare: ${avg_fare:.2f}
            - Fare volatility: {fare_volatility:.1f}%
            - Price range: ${dest_data['min_fare'].min():.0f} - ${dest_data['max_fare'].max():.0f}
            """)

    else:
        st.info("No data available for top destinations analysis")

    st.markdown("---")  # Add separator
    # Seat Availability
    st.subheader("Seat Availability")
    last_minute_query = f"""
    SELECT
        startingairport,
        destinationairport,
        AVG(seatsremaining) AS average_seat_availability,
        MIN(seatsremaining) AS min_seats_remaining,
        MAX(seatsremaining) AS max_seats_remaining,
        COUNT(*) as search_count
    FROM {FULLY_QUALIFIED_TABLE}
    WHERE {where_clause}
    GROUP BY
        startingairport,
        destinationairport
    ORDER BY
        average_seat_availability DESC
    LIMIT 10
    """

    last_minute_df = run_query(last_minute_query)

    if last_minute_df is not None and not last_minute_df.empty:
        # Display results
        st.dataframe(
            last_minute_df.style.background_gradient(
                subset=['average_seat_availability'],
                cmap='RdYlGn',
                vmin=1,
                vmax=10
            )
        )
    else:
        st.info(f"No routes with available seats found for {flight_end_date}")

    st.markdown("---")  # Add separator
    # Seat-Fare Inconsistency Analysis (Short-haul Routes)
    st.subheader("Seat-Fare Inconsistency Analysis (Short-haul Routes)")

    inconsistency_query = f"""
    WITH ShortHaulMetrics AS (
        SELECT 
            STARTINGAIRPORT,
            DESTINATIONAIRPORT,
            FLIGHTDATE,
            SEATSREMAINING,
            TOTALFARE,
            TOTALTRAVELDISTANCE,
            AVG(TOTALFARE) OVER (
                PARTITION BY STARTINGAIRPORT, DESTINATIONAIRPORT
            ) as avg_route_fare,
            AVG(SEATSREMAINING) OVER (
                PARTITION BY STARTINGAIRPORT, DESTINATIONAIRPORT
            ) as avg_route_seats
        FROM {FULLY_QUALIFIED_TABLE}
        WHERE 
            TOTALTRAVELDISTANCE <= 500  -- Short-haul routes
            AND {where_clause}
        ),
        RouteInconsistencies AS (
        SELECT 
            STARTINGAIRPORT,
            DESTINATIONAIRPORT,
            FLIGHTDATE,
            SEATSREMAINING,
            TOTALFARE,
            TOTALTRAVELDISTANCE,
            -- Flag high prices with high seats (potential revenue loss)
            CASE WHEN TOTALFARE > 1.5 * avg_route_fare 
                 AND SEATSREMAINING > 1.5 * avg_route_seats 
                 THEN 'High Price, Many Seats'
            -- Flag low prices with low seats (potential yield loss)     
            WHEN TOTALFARE < 0.7 * avg_route_fare 
                 AND SEATSREMAINING < 0.7 * avg_route_seats 
                 THEN 'Low Price, Few Seats'
            ELSE 'Normal'
            END as inconsistency_type
        FROM ShortHaulMetrics
    )
    SELECT 
        STARTINGAIRPORT,
        DESTINATIONAIRPORT,
        TOTALTRAVELDISTANCE as distance,
        inconsistency_type,
        COUNT(*) as occurrence_count,
        AVG(SEATSREMAINING) as avg_seats,
        AVG(TOTALFARE) as avg_fare,
        MIN(FLIGHTDATE) as earliest_date,
        MAX(FLIGHTDATE) as latest_date
    FROM RouteInconsistencies
    WHERE inconsistency_type != 'Normal'
    GROUP BY 
        STARTINGAIRPORT, DESTINATIONAIRPORT, TOTALTRAVELDISTANCE, inconsistency_type
    HAVING COUNT(*) >= 3  -- Show only persistent inconsistencies
    ORDER BY occurrence_count DESC, avg_fare DESC
    """

    inconsistency_df = run_query(inconsistency_query)

    if inconsistency_df is not None and not inconsistency_df.empty:
        # Display summary
        st.markdown("### Pricing Inconsistencies Found")
        
        # Create visualization
        fig = px.scatter(
            inconsistency_df,
            x='distance',
            y='avg_fare',
            size='occurrence_count',
            color='inconsistency_type',
            hover_data=[
                'startingairport', 
                'destinationairport', 
                'avg_seats'
            ],
            title="Seat-Fare Inconsistencies by Route Distance",
            labels={
                'distance': 'Route Distance (miles)',
                'avg_fare': 'Average Fare ($)',
                'inconsistency_type': 'Type of Inconsistency',
                'occurrence_count': 'Number of Occurrences'
            },
            color_discrete_map={
                'High Price, Many Seats': 'red',
                'Low Price, Few Seats': 'orange'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed analysis
        st.markdown("### Detailed Inconsistency Analysis")
        styled_df = inconsistency_df.style.background_gradient(
            subset=['avg_fare', 'avg_seats'],
            cmap='RdYlGn'
        )
        st.dataframe(styled_df)
        
        # Summary statistics
        st.markdown("### Key Findings")
        high_price = len(inconsistency_df[inconsistency_df['inconsistency_type'] == 'High Price, Many Seats'])
        low_price = len(inconsistency_df[inconsistency_df['inconsistency_type'] == 'Low Price, Few Seats'])
        
        st.markdown(f"""
        Found {len(inconsistency_df)} routes with pricing inconsistencies:
        - 🔴 {high_price} routes with high prices despite high seat availability
        - 🟠 {low_price} routes with low prices despite low seat availability
        """)
    else:
        st.info("No significant pricing inconsistencies found in short-haul routes.")
        
else:  # Ops Controller
    # --- 2. Query Section ---
    st.markdown("### Query Route Operations Data")
    user_query = st.text_input(
        "Enter your operational query:", 
        placeholder="Example: Show routes with less than 5 seats available from BOS"
    )
    
    if user_query:
        with st.spinner("Generating SQL..."):
            generated_sql = translate_to_sql(user_query)

            if generated_sql:
                st.markdown("### Generated SQL")
                st.code(generated_sql, language="sql")

                with st.spinner("Running query..."):
                    result_df = run_query(generated_sql)

                    if result_df is not None and not result_df.empty:
                        st.markdown("### Query Results")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Generate insights
                        st.markdown("### 🔍 Query Insights")
                        insights_generator(result_df)

    # --- 3. Route Capacity Analysis ---
    st.markdown("---")
    st.subheader("Route Capacity Analysis")
    
    capacity_query = f"""
    SELECT
        startingairport,
        destinationairport,
        AVG(seatsremaining) AS average_seat_availability,
        MIN(seatsremaining) AS min_seats_remaining,
        MAX(seatsremaining) AS max_seats_remaining,
        COUNT(*) as total_flights,
        COUNT(CASE WHEN seatsremaining < 5 THEN 1 END) as critical_capacity_count
    FROM {FULLY_QUALIFIED_TABLE}
    WHERE {where_clause}
    GROUP BY
        startingairport,
        destinationairport
    HAVING COUNT(*) >= 5
    ORDER BY
        average_seat_availability ASC
    LIMIT 15
    """
    
    capacity_df = run_query(capacity_query)
    
    if capacity_df is not None and not capacity_df.empty:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Routes Analyzed", len(capacity_df))
        with col2:
            st.metric("Average Seats Available", 
                     f"{capacity_df['average_seat_availability'].mean():.1f}")
        with col3:
            st.metric("Routes with Critical Capacity", 
                     len(capacity_df[capacity_df['average_seat_availability'] < 5]))
        
        # Capacity heatmap
        fig = px.scatter(
            capacity_df,
            x='startingairport',
            y='destinationairport',
            size='total_flights',
            color='average_seat_availability',
            hover_data=['min_seats_remaining', 'max_seats_remaining', 'critical_capacity_count'],
            title="Route Capacity Distribution",
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display detailed table
        st.markdown("### Detailed Route Analysis")
        st.dataframe(
            capacity_df.style.background_gradient(
                subset=['average_seat_availability'],
                cmap='RdYlGn'
            ),
            use_container_width=True
        )
