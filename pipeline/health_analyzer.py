import os
import sqlite3
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv



load_dotenv()


if "GROQ_API_KEY" not in os.environ:
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file:\n"
        "GROQ_API_KEY=your_key_here"
    )


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "health_data.db")


# SYSTEM PROMPT FOR AGENT-1 (SQL Generator)


SQL_AGENT_PROMPT = """You are an expert SQL query generator for health data analysis.

DATABASE SCHEMA:

Table: health_dataset_1 (2000 patients)
Columns:
- Patient_Number (INTEGER, Primary Key)
- Blood_Pressure_Abnormality (INTEGER: 0=Normal, 1=Abnormal)
- Level_of_Hemoglobin (REAL, g/dl)
- Genetic_Pedigree_Coefficient (REAL, 0-1, family history strength)
- Age (INTEGER)
- BMI (REAL)
- Sex (INTEGER: 0=Male, 1=Female)
- Pregnancy (INTEGER: 0=No, 1=Yes)
- Smoking (INTEGER: 0=No, 1=Yes)
- salt_content_in_the_diet (REAL, mg/day)
- alcohol_consumption_per_day (REAL, ml/day)
- Level_of_Stress (INTEGER: 1=Low, 2=Normal, 3=High)
- Chronic_kidney_disease (INTEGER: 0=No, 1=Yes)
- Adrenal_and_thyroid_disorders (INTEGER: 0=No, 1=Yes)

Table: health_dataset_2 (20000 records)
Columns:
- Patient_Number (INTEGER, Foreign Key)
- Day_Number (INTEGER, 1-10)
- Physical_activity (INTEGER, steps/day)

RULES:
1. Generate ONLY valid SQLite syntax
2. Use JOIN when both tables needed (ON Patient_Number)
3. Aggregate physical activity with AVG(), SUM(), etc.
4. Return ONLY the SQL query - no explanations, no markdown, no backticks
5. Use proper WHERE, GROUP BY, HAVING, ORDER BY clauses

EXAMPLES:

Q: "Average BMI of smokers?"
SQL: SELECT AVG(BMI) as avg_bmi, COUNT(*) as count FROM health_dataset_1 WHERE Smoking = 1

Q: "Patients with high stress and low activity?"
SQL: SELECT h1.Patient_Number, h1.Age, h1.BMI, h1.Level_of_Stress, AVG(h2.Physical_activity) as avg_steps FROM health_dataset_1 h1 JOIN health_dataset_2 h2 ON h1.Patient_Number = h2.Patient_Number WHERE h1.Level_of_Stress = 3 GROUP BY h1.Patient_Number HAVING AVG(h2.Physical_activity) < 5000 LIMIT 100

Q: "Female patients with kidney disease count"
SQL: SELECT COUNT(*) as count FROM health_dataset_1 WHERE Sex = 1 AND Chronic_kidney_disease = 1

Q: "Top 10 most active patients"
SQL: SELECT h1.Patient_Number, h1.Age, h1.Sex, AVG(h2.Physical_activity) as avg_steps FROM health_dataset_1 h1 JOIN health_dataset_2 h2 ON h1.Patient_Number = h2.Patient_Number GROUP BY h1.Patient_Number ORDER BY avg_steps DESC LIMIT 10

Q: "Average hemoglobin by stress level"
SQL: SELECT Level_of_Stress, AVG(Level_of_Hemoglobin) as avg_hemoglobin, COUNT(*) as count FROM health_dataset_1 GROUP BY Level_of_Stress

Now generate SQL for:
"""


# SYSTEM PROMPT FOR AGENT-2 (Insight Generator)


INSIGHT_AGENT_PROMPT = """You are a health data analyst providing insights from patient data analysis.

IMPORTANT MEDICAL DISCLAIMER:
- DO NOT provide diagnosis, treatment, or medical advice
- Only provide descriptive insights and lifestyle-level observations
- Always recommend consulting healthcare professionals for medical decisions

GUIDELINES:
1. Analyze the data and explain what it reveals
2. Provide health context in simple, clear language
3. Give lifestyle-level observations (not medical recommendations)
4. Be empathetic and evidence-based
5. Always reference the actual numbers from the results

HEALTH REFERENCE VALUES:
- BMI: <18.5 (Underweight), 18.5-24.9 (Normal), 25-29.9 (Overweight), 30+ (Obese)
- Physical Activity: <5000 steps (Sedentary), 5000-7500 (Low Active), 7500-10000 (Active), 10000+ (Very Active)
- Blood Pressure: 0=Normal, 1=Abnormal (requires medical attention)
- Stress Levels: 1=Low, 2=Normal, 3=High (high stress increases cardiovascular risk)
- Genetic Pedigree Coefficient: Close to 1 = strong family history of disease

FORMAT YOUR RESPONSE AS:

**📊 Summary:**
[Brief overview of what the data shows]

**🔍 Key Findings:**
- [Finding 1 with specific numbers]
- [Finding 2 with specific numbers]
- [Finding 3 with specific numbers]

**💡 Health Implications:**
[What this means for health and risk factors]

**✅ Observations & Suggestions:**
[2-3 lifestyle-level observations based on the data]
[Reminder to consult healthcare professionals]

Keep it concise, professional, and focused on practical insights.
NEVER diagnose or prescribe - only describe patterns in the data.
"""


# STATE DEFINITION


class PipelineState(TypedDict):
    """State passed through the pipeline"""
    user_query: str          
    sql_query: str            
    results_df: object       
    formatted_output: str     
    insights: str             
    final_response: str       
    error: str                


# STEP 1: AGENT-1 (SQL GENERATOR)


# SQL Safety Configuration
FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", "REPLACE"]

class Agent1_SQLGenerator:
   
    
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,  
            max_tokens=500
        )
    
    def __call__(self, state: PipelineState) -> PipelineState:
       
        print("STEP 1: AGENT-1 (SQL GENERATOR)")
        print(f"User Query: {state['user_query']}\n")
        
        
        messages = [
            SystemMessage(content=SQL_AGENT_PROMPT),
            HumanMessage(content=state['user_query'])
        ]
        
        response = self.llm.invoke(messages)
        sql_query = response.content.strip()
        
       
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        
        if any(word in sql_query.upper() for word in FORBIDDEN_KEYWORDS):
            state["error"] = " Unsafe SQL operation detected. Only SELECT queries are allowed."
            state['sql_query'] = sql_query
            print(f" SECURITY ALERT: Unsafe SQL detected!\n{sql_query}\n")
            print(" Query blocked for safety\n")
            return state
        
        state['sql_query'] = sql_query
        
        print(f"Generated SQL:\n{sql_query}\n")
        print(" Agent-1 Complete\n")
        
        return state


# STEP 2: SQL EXECUTOR (SQLite + Pandas)


class SQLExecutor:
    
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def __call__(self, state: PipelineState) -> PipelineState:
       
        print("STEP 2: SQL EXECUTION (SQLite)")
       
        try:
            
            conn = sqlite3.connect(self.db_path)
            
            
            df = pd.read_sql_query(state['sql_query'], conn)
            conn.close()
            
            
            state['results_df'] = df
            
           
            if len(df) > 0:
                formatted = f"Query returned {len(df)} row(s)\n\n"
                formatted += "RESULTS:\n"
                formatted += "=" * 70 + "\n"
                formatted += df.to_string(index=False)
                formatted += "\n" + "=" * 70
                
                
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0 and len(df) > 1:
                    formatted += "\n\nSUMMARY STATISTICS:\n"
                    formatted += "=" * 70 + "\n"
                    formatted += df[numeric_cols].describe().to_string()
                
                state['formatted_output'] = formatted
            else:
                state['formatted_output'] = "No results found."
            
            print(f"Execution Status: SUCCESS")
            print(f"Rows returned: {len(df)}\n")
            print(state['formatted_output'])
            print("\nSQL Execution Complete\n")
            
        except Exception as e:
            error_msg = f"SQL Execution Error: {str(e)}"
            state['error'] = error_msg
            state['formatted_output'] = f"{error_msg}"
            print(f"\nError: {error_msg}\n")
        
        return state


# STEP 3: AGENT-2 (INSIGHT GENERATOR)


class Agent2_InsightGenerator:
    
    
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",  
            temperature=0.7, 
            max_tokens=1000
        )
    
    def __call__(self, state: PipelineState) -> PipelineState:
        
        print("STEP 3: AGENT-2 (INSIGHT GEN)")
       
        
        
        if state.get('error'):
            state['insights'] = f"⚠️ Unable to generate insights due to error:\n{state['error']}"
            print(f"Skipping due to error\n")
            return state
        
        
        df = state['results_df']
        
        
        data_summary = f"Query returned {len(df)} row(s)\n\n"
        
        if len(df) > 0:
           
            data_summary += "SAMPLE DATA (first few rows):\n"
            data_summary += df.head(5).to_string(index=False) + "\n\n"
            
          
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                data_summary += "SUMMARY STATISTICS:\n"
                data_summary += df[numeric_cols].describe().to_string()
            else:
                
                data_summary += "VALUE DISTRIBUTION:\n"
                for col in df.columns[:3]:  
                    if len(df[col].unique()) <= 10:
                        data_summary += f"\n{col}:\n{df[col].value_counts().to_string()}\n"
        else:
            data_summary = "No results found."
        
        analysis_prompt = f"""
USER'S ORIGINAL QUESTION:
{state['user_query']}

SQL QUERY EXECUTED:
{state['sql_query']}

DATA SUMMARY:
{data_summary}

Based on this data, provide descriptive insights and lifestyle-level observations.
"""
        
        print("Generating insights from results...\n")
        
        messages = [
            SystemMessage(content=INSIGHT_AGENT_PROMPT),
            HumanMessage(content=analysis_prompt)
        ]
        
        response = self.llm.invoke(messages)
        state['insights'] = response.content
        
        print("Generated Insights:")
        print("-" * 70)
        print(state['insights'])
        print("-" * 70)
        print("\nAgent-2 Complete\n")
        
        return state


# STEP 4: FORMAT FINAL USER RESPONSE


class FinalResponseFormatter:
    
    
    def __call__(self, state: PipelineState) -> PipelineState:
        print("STEP 4: FINAL USER RESPONSE")
        
        # Build complete user response
        final_response = f"""
 YOUR QUESTION:
{state['user_query']}

SQL QUERY GENERATED:
{state['sql_query']}

{state['formatted_output']}

 HEALTH INSIGHTS & RECOMMENDATIONS:

{state['insights']}


"""
        
        state['final_response'] = final_response
        
        print("Final Response Prepared\n")
        
        return state


# LANGGRAPH PIPELINE


def create_pipeline(db_path: str = DB_PATH):
    
    
    # Initialize components
    agent1 = Agent1_SQLGenerator()
    sql_executor = SQLExecutor(db_path)
    agent2 = Agent2_InsightGenerator()
    formatter = FinalResponseFormatter()
    
    # Create workflow graph
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("agent1_sql_gen", agent1)
    workflow.add_node("sql_executor", sql_executor)
    workflow.add_node("agent2_insights", agent2)
    workflow.add_node("final_formatter", formatter)
    
    # Define linear flow
    workflow.set_entry_point("agent1_sql_gen")
    workflow.add_edge("agent1_sql_gen", "sql_executor")
    workflow.add_edge("sql_executor", "agent2_insights")
    workflow.add_edge("agent2_insights", "final_formatter")
    workflow.add_edge("final_formatter", END)
    
    return workflow.compile()


# MAIN FUNCTION


def query_health_data(user_query: str, db_path: str = DB_PATH):
    
    
    print("\n" + "=" * 70)
    print("HEALTH DATA ANALYSIS - COMPLETE AGENTIC PIPELINE")
    print("=" * 70)
    print(f"\nUSER QUERY: {user_query}\n")
    print("=" * 70 + "\n")
    
    # Create pipeline
    pipeline = create_pipeline(db_path)
    
    # Initialize state
    initial_state = {
        "user_query": user_query,
        "sql_query": "",
        "results_df": None,
        "formatted_output": "",
        "insights": "",
        "final_response": "",
        "error": ""
    }
    
    # Run complete pipeline
    final_state = pipeline.invoke(initial_state)
    
    # Display final user response
    print("\n" + "=" * 70)
    print("COMPLETE USER RESPONSE")
    print("=" * 70)
    print(final_state['final_response'])
    print("=" * 70 + "\n")
    
    return final_state

# TESTING


if __name__ == "__main__":
    
    # Test queries
    test_queries = [
        "What is the average BMI of smokers?",
        "How many female patients have abnormal blood pressure?",
        "Show me the top 10 patients with highest physical activity",
        "What is the average hemoglobin level for each stress category?",
        "Find patients who smoke, have high stress, and BMI over 30"
    ]
    
    # Run first query as test
    result = query_health_data(test_queries[0])
    