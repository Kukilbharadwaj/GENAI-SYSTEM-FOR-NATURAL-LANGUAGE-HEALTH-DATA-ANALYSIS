import streamlit as st
import pandas as pd
import sys
import os
import sqlite3
from datetime import datetime
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.health_analyzer import query_health_data, DB_PATH

# Initialize logging table
def init_logging_table():
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS app_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_query TEXT,
                generated_sql TEXT,
                result_count INTEGER,
                error_message TEXT,
                execution_time_ms REAL,
                session_id TEXT
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON app_logs(timestamp)
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not initialize logging table: {e}")

def log_query(user_query, generated_sql, result_count, error_message, execution_time_ms, session_id):
    """Log query details to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO app_logs 
            (timestamp, user_query, generated_sql, result_count, error_message, execution_time_ms, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            user_query,
            generated_sql,
            result_count,
            error_message,
            execution_time_ms,
            session_id
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not log query: {e}")

# Page configuration
st.set_page_config(
    page_title="Health Data Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 1.05rem;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    /* Expander styling - visible in all states */
    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    div[data-testid="stExpander"]:hover {
        border-color: #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Expander header - always visible */
    div[data-testid="stExpander"] summary {
        background-color: #f0f2f6;
        color: #0e1117 !important;
        font-weight: 500;
        padding: 0.75rem 1rem;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] summary:hover {
        background-color: #e6e9ef;
        color: #1f77b4 !important;
    }
    /* Expander content - visible when open */
    div[data-testid="stExpander"] div[role="region"] {
        background-color: #ffffff;
        color: #0e1117;
        padding: 1rem;
    }
    div[data-testid="stExpander"] p, 
    div[data-testid="stExpander"] span,
    div[data-testid="stExpander"] li {
        color: #0e1117 !important;
    }
    /* SQL code block styling */
    div[data-testid="stExpander"] code {
        background-color: #f0f2f6 !important;
        color: #d73a49 !important;
    }
    div[data-testid="stExpander"] pre {
        background-color: #f6f8fa !important;
    }
    /* Dataframe styling in expanders */
    div[data-testid="stExpander"] [data-testid="stDataFrame"],
    div[data-testid="stExpander"] table {
        background-color: white;
    }
    div[data-testid="stExpander"] table th {
        background-color: #f0f2f6 !important;
        color: #0e1117 !important;
        font-weight: 600;
    }
    div[data-testid="stExpander"] table td {
        color: #0e1117 !important;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header">Health Data Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered health data insights from natural language queries</p>', unsafe_allow_html=True)


init_logging_table()


if 'history' not in st.session_state:
    st.session_state.history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')


with st.container():
    st.markdown("###  Ask a question about health data")

   
    st.info("**Safe Query Environment** • Only read-only SELECT queries are allowed")

    # Example queries in an expander
    with st.expander(" Example Questions"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - What is the average BMI of smokers?
            - How many female patients have abnormal blood pressure?
            - Show me the top 10 patients with highest physical activity
            """)
        with col2:
            st.markdown("""
            - What is the average hemoglobin level for each stress category?
            - Find patients who smoke, have high stress, and BMI over 30
            - Compare physical activity between smokers and non-smokers
            """)

    # Query input
    user_query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What is the average BMI of smokers?",
        height=100,
        label_visibility="collapsed"
    )

    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("Analyze", use_container_width=True, type="primary")

# Process query
if analyze_button and user_query.strip():
    with st.spinner(" Analyzing your query..."):
        start_time = datetime.now()
        result = None
        error_msg = None
        
        try:
            # Run the analysis
            result = query_health_data(user_query, DB_PATH)

            # Check for errors (including safety errors)
            if result.get('error'):
                error_msg = result['error']
                st.error(f" {result['error']}")
                if 'Unsafe SQL' in result['error']:
                    st.warning(" **Security Notice:** This system only allows safe, read-only queries to protect data integrity.")
                # Continue execution instead of returning

            # Display results
            st.markdown("---")
            st.markdown("###  Results")

            # SQL Query in expander
            with st.expander(" Generated SQL Query", expanded=False):
                st.code(result['sql_query'], language='sql')

            # Data Results
            if result['results_df'] is not None and not result['results_df'].empty:
                st.markdown("####  Data")

                # Display dataframe
                st.dataframe(
                    result['results_df'],
                    use_container_width=True,
                    hide_index=True
                )

                # Summary statistics if available
                numeric_cols = result['results_df'].select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0 and len(result['results_df']) > 1:
                    with st.expander("  Summary Statistics"):
                        st.dataframe(
                            result['results_df'][numeric_cols].describe(),
                            use_container_width=True
                        )

            # Insights
            if result['insights']:
                st.markdown("####  Health Insights")
                st.info("ℹ **Disclaimer:** These are data observations only. Not medical advice. Consult healthcare professionals for medical decisions.")
                st.markdown(result['insights'])

            # Add to history
            st.session_state.history.append({
                'query': user_query,
                'result': result
            })

        except Exception as e:
            error_msg = str(e)
            st.error(f" An error occurred: {str(e)}")
            st.info(" Try rephrasing your question or check one of the examples above.")
        
        finally:
            # Log the query
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result_count = len(result['results_df']) if result and result.get('results_df') is not None else 0
            generated_sql = result.get('sql_query', '') if result else ''
            
            log_query(
                user_query=user_query,
                generated_sql=generated_sql,
                result_count=result_count,
                error_message=error_msg,
                execution_time_ms=execution_time,
                session_id=st.session_state.session_id
            )

elif analyze_button:
    st.warning(" Please enter a question first.")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9rem;">Powered by AI • Built with Streamlit</p>',
    unsafe_allow_html=True
)