# GenAI Health Data Analysis

An intelligent health data analysis system powered by LangGraph and Groq LLM. This project features an agentic pipeline that converts natural language queries into SQL, executes them, and generates actionable insights from health datasets.

## 🏗️ Project Structure

```
GenAI-Health-Data-Analysis/
│
├── app/
│   └── app.py                      # Streamlit UI
│
├── pipeline/
│   └── health_analyzer.py          # Agentic LangGraph pipeline
│
├── evaluation/
│   ├── evaluation_framework.py     # Evaluation metrics and test cases
│   └── run_evaluation.py           # Evaluation runner script
│
├── data/
│   ├── raw/                        # Original datasets
│   ├── processed/                  # Processed data
│   └── health_data.db              # SQLite database
│
├── preprocess/
│   └── data_preprocessing.py       # Data preprocessing utilities
│
├── .env.example                    # Environment variables template
├── .gitignore
├── requirements.txt
└── README.md
```

## ✨ Features

- **Natural Language Queries**: Ask health-related questions in plain English
- **Agentic LangGraph Pipeline**: Multi-agent system with specialized roles:
  - 🔍 SQL Generator Agent
  - ⚙️ Query Executor Agent
  - 💡 Insight Generator Agent
  - ✅ Validator Agent
- **Streamlit UI**: Clean, modern interface for data exploration
- **Evaluation Framework**: Comprehensive testing and quality metrics

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd healthcare
```

### 2. Set Up Python Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
copy .env.example .env
```

Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=your_actual_api_key_here
```

### 5. Preprocess Data (First Time Only)

```bash
python Preprocess/data_preprocessing.py
```

This will:
- Load datasets from `data/raw/`
- Clean and validate data
- Create SQLite database at `data/health_data.db`

### 6. Run the Application

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Usage Examples

Try these queries in the Streamlit interface:

- "What is the average BMI of smokers?"
- "How many patients have high stress levels?"
- "Show correlation between alcohol consumption and blood pressure"
- "Which age group has the highest prevalence of chronic kidney disease?"

## 🧪 Running Evaluations

Evaluate the pipeline performance:

```bash
python evaluation/run_evaluation.py
```

This generates:
- SQL quality metrics
- Execution success rates
- Insight generation scores
- Detailed report with timestamp

## 🗄️ Database Schema

### health_dataset_1 (2000 patients)
- Patient demographics (Age, Sex, BMI)
- Health conditions (Blood Pressure, Hemoglobin, Chronic diseases)
- Lifestyle factors (Smoking, Alcohol, Diet, Stress)

### health_dataset_2 (2000 patients)
- Physical activity levels
- Sleep patterns
- Additional health metrics

## 🛠️ Technologies

- **LangGraph**: Agentic workflow orchestration
- **Groq**: High-speed LLM inference
- **Streamlit**: Web UI framework
- **Pandas**: Data manipulation
- **SQLite**: Database storage
- **Python-dotenv**: Environment management

## 📝 Development

### Project Scripts

- `preprocess/data_preprocessing.py`: Process raw Excel files into SQLite DB
- `pipeline/health_analyzer.py`: Core agentic pipeline logic
- `evaluation/evaluation_framework.py`: Testing framework
- `app/app.py`: Streamlit application

### Adding New Features

1. Modify the pipeline in `pipeline/health_analyzer.py`
2. Update UI components in `app/app.py`
3. Add test cases to `evaluation/evaluation_framework.py`
4. Run evaluations to verify changes

## 🔐 Security Notes

- Use `.env.example` as a template


