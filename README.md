# Multi-Head RAG Strategy

## Overview

The Multi-Head RAG Strategy is an advanced method for retrieving documents relevant to a set of query embeddings from a database of document embeddings. This strategy employs the concept of multiple attention heads, a fundamental aspect of transformer models, to evaluate the relevance of documents in a nuanced and effective manner.

By using embeddings from different layers and attention heads, the strategy can capture diverse aspects of semantic similarity between queries and documents, leading to more accurate retrieval results.

## Components

The system comprises several key components that work together to enable efficient and accurate document retrieval:

### article_emd.py

	•	Purpose: Extracts single-aspect embeddings for articles from the last h layers of the embedding model.
	•	Model Used: Utilizes the BAAI/bge-large-en-v1.5 embedding model to generate high-quality embeddings that capture the semantic content of articles.

### query_emd.py

	•	Purpose: Extracts single-aspect embeddings for queries from the last h layers.
	•	Note: It is crucial to use the same embedding model as in article_emd.py to ensure consistency in the embeddings’ semantic space.

### db_store.py

	•	VectorDB Class: Provides an abstraction for a vector database that stores document embeddings. It includes functionalities for:
	•	Adding Articles: Stores the embeddings of articles for efficient retrieval.
	•	Attention-Based Search: Searches through embeddings based on different layers and attention heads to find the most relevant documents.

### retrieval.py

	•	MultiHeadStrategy Class: Implements the Strategy interface using the Multi-Head RAG strategy. It evaluates query embeddings by considering multiple attention heads, which helps in capturing complex patterns of relevance between queries and documents.

## cMain Functionality

get_retrieved_chunks Function

The get_retrieved_chunks function is the core of the document retrieval system. It leverages embedding similarity to retrieve the top documents for a set of queries using the MultiHeadStrategy.

Parameters

	•	data (List[str]): A list of article texts to be embedded and stored in the database.
	•	questions (List[str]): A list of query texts for which relevant documents are to be retrieved.
	•	distance_metric (DistanceMetric): The metric used to measure distance between embeddings (default is DistanceMetric.COSINE).
	•	top_n (int): Number of top documents to retrieve for each query (default is 5).
	•	layers (Union[int, List[int]]): Specific layer(s) from which embeddings are extracted (default is 23).

Returns

	•	List[List[Article]]: A list of lists containing the top articles for each query, ranked by relevance.

Usage

To utilize the Multi-Head RAG Strategy and the get_retrieved_chunks function, follow these steps:

	1.	Prepare Your Data: Define a list of article texts and corresponding queries that you wish to analyze.
	2.	Initialize the VectorDB: Create an instance of VectorDB with the desired distance metric to store and manage document embeddings.
	3.	Generate Embeddings: Use the generate_embeddings function from article_emd.py and the generate_query_embeddings function from query_emd.py to create embeddings for articles and queries, respectively.
	4.	Instantiate the Strategy: Create an instance of MultiHeadStrategy with appropriate parameters, such as the name of the strategy, the vector database instance, the layer for embeddings, and the weight function.
	5.	Execute the Retrieval: Call the get_retrieved_chunks function with your data and queries to obtain the top relevant documents for each query.

 
