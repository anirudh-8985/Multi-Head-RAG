from typing import List, Union

from article_emd import (
    Article,
    generate_embeddings
)

from db_store import (
    VectorDB,
    DistanceMetric
)

from query_embed import (
    Query,
    generate_query_embeddings
)

from retrieval import (
    MultiHeadStrategy,
    weight_function
)

def get_retrieved_chunks(
    data: List[str], 
    questions: List[str],
    distance_metric: DistanceMetric = DistanceMetric.COSINE,
    top_n: int = 5,
    layers: Union[int, List[int]] = 23
) -> List[List[Article]]:
    """
    Retrieves top documents for given questions using embedding similarity.

    Args:
        data (List[str]): List of article texts.
        questions (List[str]): List of query texts.
        distance_metric (DistanceMetric): The metric used to measure distance.
        top_n (int): Number of top documents to retrieve.
        layers (Union[int, List[int]]): Specific layer(s) for embeddings.

    Returns:
        List[List[Article]]: A list of lists containing top articles for each query.
    """
    
    # Convert input data and questions into Article and Query objects
    articles = [Article(text) for text in data]
    queries = [Query(text) for text in questions]
    
    # Generate embeddings for articles and queries
    article_embeddings = generate_embeddings(articles, layers)
    query_embeddings = generate_query_embeddings(queries, layers)
    
    # Initialize vector database and add article embeddings
    db = VectorDB(distance_metric=distance_metric)
    db.add_articles(article_embeddings)
    
    # Initialize the multi-head retrieval strategy
    strategy = MultiHeadStrategy(name="MultiHead", db=db, layer=layers, weight_fn=weight_function)
    
    # Retrieve top documents for each query embedding
    top_documents = [strategy._get_picks(query_embedding, top_n) for query_embedding in query_embeddings]
    
    return top_documents

# Example data and queries
data = ['text1', 'text2', 'text3']
questions = ['Query1', 'Query2', 'Query3']

# Retrieve top documents
top_docs = get_retrieved_chunks(data, questions)