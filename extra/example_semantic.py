from rlm.rlm_repl import RLM_REPL

def main():
    """
    Example that FORCES sub-LLM calls by requiring semantic analysis.
    This demonstrates async batch processing with llm_batch().
    """
    print("Example: Multi-document semantic analysis (forces sub-LLM usage)\n")
    
    # Create a context with multiple "documents" that require understanding
    # Using triple newlines to make splitting easier
    context = """
DOCUMENT 1 - Financial Report:
The quarterly revenue increased by 23% reaching $4.5M. Operating costs were reduced by 15%.
The company shows strong growth potential with expanding market share in Asia.


DOCUMENT 2 - Customer Feedback:
Customers complain about slow delivery times and poor customer service response.
Many users mentioned the product quality is excellent but support needs improvement.


DOCUMENT 3 - Technical Analysis:
The new feature implementation caused a 40% increase in server load.
Database queries are taking 3x longer than expected. Performance optimization needed urgently.


DOCUMENT 4 - Marketing Campaign:
Social media engagement increased by 200%. Brand awareness is at an all-time high.
Influencer partnerships are driving significant traffic to the website.


DOCUMENT 5 - Product Development:
Three major features completed ahead of schedule. Bug count reduced by 60%.
The development team has shown exceptional productivity this quarter.


DOCUMENT 6 - HR Report:
Employee satisfaction scores dropped by 15%. High turnover rate in engineering department.
Compensation concerns raised in anonymous surveys. Work-life balance issues reported.
"""
    
    # Query that requires understanding each section
    query = """
    Analyze all documents and provide:
    1. Overall company health score (1-10)
    2. Top 3 priorities that need immediate attention
    3. Top 3 positive achievements
    
    IMPORTANT: Documents are separated by blank lines. Split the context into 6 separate 
    documents and process each one in parallel using llm_batch() for efficiency.
    Each document starts with "DOCUMENT N -".
    """
    
    # Initialize RLM
    rlm = RLM_REPL(
        model="gpt-5-mini",                    # Root LLM for orchestration
        recursive_model="gpt-5-mini",  # Cheaper sub-LLMs for analysis
        enable_logging=True,
        max_iterations=5
    )
    
    print(f"Query: {query}\n")
    print("=" * 80)
    result = rlm.completion(context=context, query=query)
    
    print("\n" + "=" * 80)
    print(f"\nFinal Answer:\n{result}\n")

if __name__ == "__main__":
    main()