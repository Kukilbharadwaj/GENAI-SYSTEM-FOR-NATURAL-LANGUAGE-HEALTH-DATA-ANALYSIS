
import time
import pandas as pd
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.health_analyzer import query_health_data, DB_PATH
import sqlite3
from datetime import datetime


# TEST CASES
EVALUATION_TEST_CASES = [
    {
        "id": 1,
        "category": "Basic Aggregation",
        "query": "What is the average BMI of smokers?",
        "expected_sql_keywords": ["AVG", "BMI", "Smoking", "WHERE"],
        "expected_tables": ["health_dataset_1"],
        "complexity": "simple",
        "should_succeed": True
    },
    {
        "id": 2,
        "category": "Filtering & Counting",
        "query": "How many female patients have chronic kidney disease?",
        "expected_sql_keywords": ["COUNT", "Sex", "Chronic_kidney_disease", "WHERE"],
        "expected_tables": ["health_dataset_1"],
        "complexity": "simple",
        "should_succeed": True
    },
    {
        "id": 3,
        "category": "JOIN Query",
        "query": "Show top 10 most active patients with their age and BMI",
        "expected_sql_keywords": ["JOIN", "Physical_activity", "ORDER BY", "LIMIT"],
        "expected_tables": ["health_dataset_1", "health_dataset_2"],
        "complexity": "medium",
        "should_succeed": True
    },
    {
        "id": 4,
        "category": "Grouping & Aggregation",
        "query": "What is average hemoglobin level for each stress category?",
        "expected_sql_keywords": ["AVG", "Level_of_Hemoglobin", "Level_of_Stress", "GROUP BY"],
        "expected_tables": ["health_dataset_1"],
        "complexity": "medium",
        "should_succeed": True
    },
    {
        "id": 5,
        "category": "Complex Filtering",
        "query": "Find patients with high stress, abnormal blood pressure, and low physical activity",
        "expected_sql_keywords": ["JOIN", "WHERE", "Level_of_Stress", "Blood_Pressure_Abnormality", "Physical_activity"],
        "expected_tables": ["health_dataset_1", "health_dataset_2"],
        "complexity": "complex",
        "should_succeed": True
    },
    {
        "id": 6,
        "category": "Multiple Conditions",
        "query": "Show smokers over 50 with BMI greater than 30",
        "expected_sql_keywords": ["WHERE", "Smoking", "Age", "BMI"],
        "expected_tables": ["health_dataset_1"],
        "complexity": "medium",
        "should_succeed": True
    },
    {
        "id": 7,
        "category": "Advanced JOIN",
        "query": "Average daily steps for patients with thyroid disorders",
        "expected_sql_keywords": ["JOIN", "AVG", "Physical_activity", "Adrenal_and_thyroid_disorders"],
        "expected_tables": ["health_dataset_1", "health_dataset_2"],
        "complexity": "medium",
        "should_succeed": True
    },
    {
        "id": 8,
        "category": "Correlation Analysis",
        "query": "Compare average alcohol consumption between patients with and without kidney disease",
        "expected_sql_keywords": ["AVG", "alcohol_consumption_per_day", "Chronic_kidney_disease", "GROUP BY"],
        "expected_tables": ["health_dataset_1"],
        "complexity": "medium",
        "should_succeed": True
    },
    {
        "id": 9,
        "category": "Complex Aggregation",
        "query": "What percentage of pregnant women have abnormal blood pressure?",
        "expected_sql_keywords": ["COUNT", "Pregnancy", "Blood_Pressure_Abnormality"],
        "expected_tables": ["health_dataset_1"],
        "complexity": "medium",
        "should_succeed": True
    },
    {
        "id": 10,
        "category": "Activity Analysis",
        "query": "Find patients averaging less than 5000 steps who also smoke",
        "expected_sql_keywords": ["JOIN", "AVG", "Physical_activity", "Smoking", "HAVING"],
        "expected_tables": ["health_dataset_1", "health_dataset_2"],
        "complexity": "complex",
        "should_succeed": True
    }
]


# EVALUATION METRICS

class PipelineEvaluator:
    """
    Evaluates the complete health data analysis pipeline
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def validate_sql(self, sql_query: str, expected_keywords: List[str], expected_tables: List[str]) -> Dict:
        """
        Validate SQL query quality
        
        Returns:
            Dict with validation metrics
        """
        sql_upper = sql_query.upper()
        
        # Check for expected keywords
        keywords_found = [kw for kw in expected_keywords if kw.upper() in sql_upper]
        keyword_score = len(keywords_found) / len(expected_keywords) if expected_keywords else 1.0
        
        # Check for expected tables
        tables_found = [tbl for tbl in expected_tables if tbl in sql_query]
        table_score = len(tables_found) / len(expected_tables) if expected_tables else 1.0
        
        # Check SQL starts with SELECT
        is_select = sql_query.strip().upper().startswith("SELECT")
        
        # Check for dangerous operations
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
        is_safe = not any(word in sql_upper for word in forbidden)
        
        # Check basic syntax
        has_from = "FROM" in sql_upper
        
        return {
            "is_select": is_select,
            "is_safe": is_safe,
            "has_from": has_from,
            "keyword_score": keyword_score,
            "keywords_found": keywords_found,
            "keywords_missing": [kw for kw in expected_keywords if kw.upper() not in sql_upper],
            "table_score": table_score,
            "tables_found": tables_found,
            "syntax_valid": is_select and has_from,
            "overall_sql_quality": (keyword_score + table_score) / 2
        }
    
    def check_execution(self, sql_query: str) -> Dict:
        """
        Test if SQL executes successfully
        
        Returns:
            Dict with execution metrics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            return {
                "execution_success": True,
                "row_count": len(df),
                "column_count": len(df.columns),
                "has_results": len(df) > 0,
                "error": None
            }
        except Exception as e:
            return {
                "execution_success": False,
                "row_count": 0,
                "column_count": 0,
                "has_results": False,
                "error": str(e)
            }
    
    def evaluate_insights(self, insights: str, user_query: str) -> Dict:
        """
        Evaluate quality of generated insights
        
        Returns:
            Dict with insight quality metrics
        """
        if not insights or insights.strip() == "":
            return {
                "has_insights": False,
                "length": 0,
                "has_summary": False,
                "has_findings": False,
                "has_suggestions": False,
                "quality_score": 0.0
            }
        
        insights_lower = insights.lower()
        
        # Check for key sections
        has_summary = "summary" in insights_lower or "📊" in insights
        has_findings = "finding" in insights_lower or "🔍" in insights
        has_suggestions = "suggestion" in insights_lower or "observation" in insights_lower or "✅" in insights
        has_health_context = any(word in insights_lower for word in ["bmi", "health", "risk", "patient", "activity", "stress"])
        
        # Calculate quality score
        quality_components = [
            has_summary,
            has_findings,
            has_suggestions,
            has_health_context,
            len(insights) > 100  
        ]
        quality_score = sum(quality_components) / len(quality_components)
        
        return {
            "has_insights": True,
            "length": len(insights),
            "has_summary": has_summary,
            "has_findings": has_findings,
            "has_suggestions": has_suggestions,
            "has_health_context": has_health_context,
            "quality_score": quality_score
        }
    
    def evaluate_single_query(self, test_case: Dict) -> Dict:
       
        print(f"\n{'='*70}")
        print(f"TEST CASE #{test_case['id']}: {test_case['category']}")
        print(f"Query: {test_case['query']}")
        print(f"Complexity: {test_case['complexity'].upper()}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            # Run query through pipeline
            state = query_health_data(test_case['query'], self.db_path)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extract components
            sql_query = state.get('sql_query', '')
            insights = state.get('insights', '')
            error = state.get('error', '')
            
            # Validate SQL
            sql_metrics = self.validate_sql(
                sql_query,
                test_case['expected_sql_keywords'],
                test_case['expected_tables']
            )
            
            # Check execution
            exec_metrics = self.check_execution(sql_query)
            
            # Evaluate insights
            insight_metrics = self.evaluate_insights(insights, test_case['query'])
            
            # Overall success
            overall_success = (
                sql_metrics['syntax_valid'] and
                sql_metrics['is_safe'] and
                exec_metrics['execution_success'] and
                insight_metrics['has_insights']
            )
            
            result = {
                "test_id": test_case['id'],
                "category": test_case['category'],
                "query": test_case['query'],
                "complexity": test_case['complexity'],
                "response_time": response_time,
                "overall_success": overall_success,
                "sql_generated": sql_query,
                "sql_metrics": sql_metrics,
                "execution_metrics": exec_metrics,
                "insight_metrics": insight_metrics,
                "error": error,
                "timestamp": datetime.now().isoformat()
            }
            
            # Print summary
            print(f"\n RESULTS:")
            print(f"   Response Time: {response_time:.2f}s")
            print(f"   SQL Quality: {sql_metrics['overall_sql_quality']*100:.1f}%")
            print(f"   Execution: {'SUCCESS' if exec_metrics['execution_success'] else 'FAILED'}")
            print(f"   Insights Quality: {insight_metrics['quality_score']*100:.1f}%")
            print(f"   Overall: {'PASS' if overall_success else 'FAIL'}")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"\nEXCEPTION: {str(e)}")
            
            return {
                "test_id": test_case['id'],
                "category": test_case['category'],
                "query": test_case['query'],
                "complexity": test_case['complexity'],
                "response_time": response_time,
                "overall_success": False,
                "sql_generated": "",
                "sql_metrics": {},
                "execution_metrics": {"execution_success": False, "error": str(e)},
                "insight_metrics": {"has_insights": False},
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_evaluation(self, test_cases: List[Dict] = None, quick_mode: bool = False) -> Dict:
       
        if test_cases is None:
            test_cases = EVALUATION_TEST_CASES
        
        if quick_mode:
            test_cases = test_cases[:3]
            print("\nQUICK MODE: Running first 3 test cases only\n")
        
        self.start_time = time.time()
        self.results = []
        
        print("\n" + "="*70)
        print("STARTING PIPELINE EVALUATION")
        print(f"Total Test Cases: {len(test_cases)}")
        print("="*70)
        
        # Run each test case
        for test_case in test_cases:
            result = self.evaluate_single_query(test_case)
            self.results.append(result)
        
        self.end_time = time.time()
        
        # Generate summary report
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        
        if not self.results:
            return {"error": "No results to report"}
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['overall_success'])
        failed_tests = total_tests - successful_tests
        
        # SQL metrics
        sql_success = sum(1 for r in self.results if r.get('sql_metrics', {}).get('syntax_valid', False))
        avg_sql_quality = sum(r.get('sql_metrics', {}).get('overall_sql_quality', 0) for r in self.results) / total_tests
        
        # Execution metrics
        exec_success = sum(1 for r in self.results if r.get('execution_metrics', {}).get('execution_success', False))
        
        # Insight metrics
        insight_success = sum(1 for r in self.results if r.get('insight_metrics', {}).get('has_insights', False))
        avg_insight_quality = sum(r.get('insight_metrics', {}).get('quality_score', 0) for r in self.results) / total_tests
        
        # Performance metrics
        response_times = [r['response_time'] for r in self.results]
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # By complexity
        complexity_breakdown = {}
        for complexity in ['simple', 'medium', 'complex']:
            complexity_results = [r for r in self.results if r['complexity'] == complexity]
            if complexity_results:
                complexity_breakdown[complexity] = {
                    "total": len(complexity_results),
                    "successful": sum(1 for r in complexity_results if r['overall_success']),
                    "success_rate": sum(1 for r in complexity_results if r['overall_success']) / len(complexity_results) * 100,
                    "avg_response_time": sum(r['response_time'] for r in complexity_results) / len(complexity_results)
                }
        
        report = {
            "evaluation_summary": {
                "total_tests": total_tests,
                "successful": successful_tests,
                "failed": failed_tests,
                "success_rate": (successful_tests / total_tests) * 100,
                "total_duration": self.end_time - self.start_time
            },
            "sql_generation": {
                "valid_sql_count": sql_success,
                "success_rate": (sql_success / total_tests) * 100,
                "avg_quality_score": avg_sql_quality * 100
            },
            "query_execution": {
                "successful_executions": exec_success,
                "success_rate": (exec_success / total_tests) * 100
            },
            "insight_generation": {
                "with_insights": insight_success,
                "success_rate": (insight_success / total_tests) * 100,
                "avg_quality_score": avg_insight_quality * 100
            },
            "performance": {
                "avg_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time
            },
            "complexity_breakdown": complexity_breakdown,
            "detailed_results": self.results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Print summary
        self.print_summary_report(report)
        
        return report
    
    def print_summary_report(self, report: Dict):
        """
        Print formatted summary report
        """
        print("\n\n" + "="*70)
        print("EVALUATION SUMMARY REPORT")
        print("="*70)
        
        summary = report['evaluation_summary']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Total Duration: {summary['total_duration']:.2f}s")
        
        sql = report['sql_generation']
        print(f"\n  SQL GENERATION:")
        print(f"   Valid SQL: {sql['valid_sql_count']}/{summary['total_tests']}")
        print(f"   Success Rate: {sql['success_rate']:.1f}%")
        print(f"   Avg Quality Score: {sql['avg_quality_score']:.1f}%")
        
        exec_metrics = report['query_execution']
        print(f"\n QUERY EXECUTION:")
        print(f"   Successful: {exec_metrics['successful_executions']}/{summary['total_tests']}")
        print(f"   Success Rate: {exec_metrics['success_rate']:.1f}%")
        
        insights = report['insight_generation']
        print(f"\n INSIGHT GENERATION:")
        print(f"   With Insights: {insights['with_insights']}/{summary['total_tests']}")
        print(f"   Success Rate: {insights['success_rate']:.1f}%")
        print(f"   Avg Quality Score: {insights['avg_quality_score']:.1f}%")
        
        perf = report['performance']
        print(f"\nPERFORMANCE:")
        print(f"   Avg Response Time: {perf['avg_response_time']:.2f}s")
        print(f"   Min Response Time: {perf['min_response_time']:.2f}s")
        print(f"   Max Response Time: {perf['max_response_time']:.2f}s")
        
        print(f"\nBY COMPLEXITY:")
        for complexity, metrics in report['complexity_breakdown'].items():
            print(f"   {complexity.upper()}: {metrics['successful']}/{metrics['total']} " +
                  f"({metrics['success_rate']:.1f}%) - Avg {metrics['avg_response_time']:.2f}s")
        
        print("\n" + "="*70)
        
        # Show failed tests if any
        failed = [r for r in report['detailed_results'] if not r['overall_success']]
        if failed:
            print(f"\nFAILED TESTS ({len(failed)}):")
            for r in failed:
                print(f"   #{r['test_id']}: {r['query']}")
                if r.get('error'):
                    print(f"      Error: {r['error']}")
        
        print("\n" + "="*70 + "\n")


# QUICK EVALUATION FUNCTION


def run_quick_evaluation():
    
    evaluator = PipelineEvaluator()
    return evaluator.run_evaluation(quick_mode=True)

def run_full_evaluation():
    
    evaluator = PipelineEvaluator()
    return evaluator.run_evaluation(quick_mode=False)


# MAIN


if __name__ == "__main__":
 
    print("Running quick evaluation...")
    report = run_quick_evaluation()
    
    print("\nTo run full evaluation, use: run_full_evaluation()")
