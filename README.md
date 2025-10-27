# Flight Analytics Dashboard with LLM Integration

This Streamlit application connects to a Snowflake data warehouse containing flight pricing data from March to September 2022.  
It enables users to explore flight pricing trends, seat availability, and route anomalies through either traditional BI visualizations or an LLM-powered natural language query interface.

## Personas

### Revenue Manager (Classical Dashboard)
Focuses on pricing trends, revenue optimization, and fare patterns.  
Includes visual analytics such as fare trends, margin opportunities, anomaly detection, and top destination summaries.

### Ops Controller (LLM-Augmented Dashboard)
Focuses on seat utilization, operational metrics, and data consistency.  
Includes a natural language–to–SQL interface for querying the Snowflake dataset directly and supports interactive insights via OpenAI API.

## Dataset Overview

The dashboard uses the following key columns:
- STARTINGAIRPORT, DESTINATIONAIRPORT — route identification  
- SEARCHDATE, FLIGHTDATE — temporal dimensions  
- BASEFARE, TOTALFARE — fare metrics  
- SEATSREMAINING — seat availability  
- ELAPSEDDAYS — days between booking and flight  
- ISNONSTOP, TOTALTRAVELDISTANCE — route characteristics  

## Research Context

This dashboard supports an experimental study comparing classical BI dashboards and LLM-augmented dashboards across two personas:
- Revenue Manager (non-LLM group)  
- Ops Controller (LLM group)  

The study evaluates decision-making performance, reasoning quality, latency, and trust.

## Requirements

- Python 3.10+  
- Streamlit  
- Snowflake Connector for Python  
- Plotly  
- OpenAI Python SDK  

## License

This project is for academic and research purposes.  
© 2025 Martin Boeckle. All rights reserved.
