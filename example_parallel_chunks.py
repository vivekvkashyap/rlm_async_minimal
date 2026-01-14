from rlm.rlm_repl import RLM_REPL

def main():
    """
    Example that DEFINITELY uses llm_batch() for parallel processing.
    Creates a large context that requires chunking and parallel analysis.
    """
    print("Example: Parallel chunk processing with llm_batch()\n")
    
    # Create a large context with multiple news articles
    articles = []
    topics = [
        ("AI", "Artificial Intelligence revolutionizes healthcare with new diagnostic tools."),
        ("Climate", "Global temperatures reach record highs, urgent action needed."),
        ("Economy", "Stock markets surge on positive employment data and strong earnings."),
        ("Space", "NASA discovers water on Mars, potential for future colonization."),
        ("Technology", "Quantum computing breakthrough enables faster drug discovery."),
        ("Politics", "New legislation passes to address housing affordability crisis."),
        ("Sports", "Olympic games showcase incredible athletic achievements and records."),
        ("Health", "New vaccine shows 95% effectiveness in clinical trials."),
        ("Education", "Online learning platforms see 300% growth in student enrollment."),
        ("Energy", "Solar power costs drop below fossil fuels for the first time."),
    ]
    
    for i, (category, headline) in enumerate(topics):
        article = f"""
        ARTICLE {i+1} - Category: {category}
        {headline}
        This article discusses the implications for the future and potential impacts on society.
        Experts believe this development will significantly change the landscape of {category}.
        The findings were published after extensive research and peer review.
        """ * 50  # Repeat to make it larger
        articles.append(article)
    
    context = "\n\n".join(articles)
    
    print(f"Total context size: {len(context):,} characters")
    print(f"Number of articles: {len(articles)}\n")
    
    # Query that REQUIRES processing each chunk separately
    query = """
    You MUST use llm_batch() to process all articles in parallel for efficiency.
    
    For each article, extract:
    1. Category
    2. Main topic
    3. Sentiment (positive/negative/neutral)
    
    Then provide a summary of all categories and their sentiments.
    
    IMPORTANT: Split the context into chunks (one per article) and use llm_batch()
    to query all chunks in parallel. This is much faster than sequential processing.
    """
    
    # Initialize RLM
    rlm = RLM_REPL(
        model="gpt-4",                    # Root LLM for orchestration
        recursive_model="gpt-3.5-turbo",  # Cheaper/faster sub-LLMs
        enable_logging=True,
        max_iterations=5
    )
    
    print("=" * 80)
    result = rlm.completion(context=context, query=query)
    
    print("\n" + "=" * 80)
    print(f"\nFinal Answer:\n{result}\n")

if __name__ == "__main__":
    main()

