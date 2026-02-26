
    # python run_evaluation.py          # Quick mode (first 3 tests)
    # python run_evaluation.py quick    # Quick mode
    # python run_evaluation.py full     # Full evaluation (all tests)
    # python run_evaluation.py save     # Full evaluation + save results to CSV


import sys
import json
import pandas as pd
from datetime import datetime
from evaluation_framework import (
    PipelineEvaluator,
    run_quick_evaluation,
    run_full_evaluation,
    EVALUATION_TEST_CASES
)

def save_results_to_csv(report: dict, filename: str = None):
   
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.csv"
    
    # Convert detailed results to DataFrame
    results_df = pd.DataFrame(report['detailed_results'])
    
    # Expand nested dictionaries
    sql_metrics_df = pd.json_normalize(results_df['sql_metrics'])
    exec_metrics_df = pd.json_normalize(results_df['execution_metrics'])
    insight_metrics_df = pd.json_normalize(results_df['insight_metrics'])
    
    # Combine all metrics
    combined_df = pd.concat([
        results_df[['test_id', 'category', 'query', 'complexity', 'response_time', 'overall_success']],
        sql_metrics_df.add_prefix('sql_'),
        exec_metrics_df.add_prefix('exec_'),
        insight_metrics_df.add_prefix('insight_')
    ], axis=1)
    
    combined_df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")
    
    # Also save summary as JSON
    summary_file = filename.replace('.csv', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'evaluation_summary': report['evaluation_summary'],
            'sql_generation': report['sql_generation'],
            'query_execution': report['query_execution'],
            'insight_generation': report['insight_generation'],
            'performance': report['performance'],
            'complexity_breakdown': report['complexity_breakdown']
        }, f, indent=2)
    print(f"✅ Summary saved to: {summary_file}")

def main():
    """
    Main execution function
    """
    # Default to quick mode
    mode = "quick"
    save_results = False
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['full', 'all', 'complete']:
            mode = "full"
        elif arg in ['save', 'export']:
            mode = "full"
            save_results = True
        elif arg in ['quick', 'fast', 'q']:
            mode = "quick"
    
    # Print header
    print("\n" + "="*70)
    print("HEALTH DATA ANALYSIS PIPELINE EVALUATION")
    print("="*70)
    print(f"Mode: {mode.upper()}")
    if save_results:
        print("Results will be saved to CSV")
    print("="*70 + "\n")
    
    # Run evaluation
    if mode == "quick":
        print(f"Running QUICK evaluation ({3} test cases)...\n")
        report = run_quick_evaluation()
    else:
        print(f"Running FULL evaluation ({len(EVALUATION_TEST_CASES)} test cases)...\n")
        report = run_full_evaluation()
    
    # Save results if requested
    if save_results:
        save_results_to_csv(report)
    
    # Print final success rate
    success_rate = report['evaluation_summary']['success_rate']
    print("\n" + "="*70)
    if success_rate >= 90:
        print("EXCELLENT PERFORMANCE!")
    elif success_rate >= 75:
        print("GOOD PERFORMANCE")
    elif success_rate >= 50:
        print("ACCEPTABLE PERFORMANCE - NEEDS IMPROVEMENT")
    else:
        print("POOR PERFORMANCE - REQUIRES ATTENTION")
    print(f"Overall Success Rate: {success_rate:.1f}%")
    print("="*70 + "\n")
    
    return report

if __name__ == "__main__":
    try:
        report = main()
        
        # Exit with appropriate code
        success_rate = report['evaluation_summary']['success_rate']
        if success_rate >= 75:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except KeyboardInterrupt:
        print("\n\n  Evaluation interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
