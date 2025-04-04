# SHL Assessment Recommendation System: Technical Approach

## Problem Overview
The task was to build an intelligent recommendation system that helps hiring managers find the right SHL assessments based on natural language queries or job descriptions.

## Technical Approach

### 1. Data Collection and Processing
- **Web Scraping**: Used BeautifulSoup4 to extract assessment information from the SHL product catalog
- **Data Structure**: Created a structured representation of each assessment with name, URL, description, duration, etc.
- **Fallback Mechanism**: Implemented sample data generation for cases where web scraping fails

### 2. Vector Embeddings for Semantic Search
- **Model Selection**: Used SentenceBERT's "all-MiniLM-L6-v2" model for efficient text embeddings
- **Text Preparation**: Combined assessment attributes into a rich "search_text" field for more accurate embeddings
- **Normalization**: Applied L2 normalization to embeddings for improved cosine similarity matching

### 3. Similarity Search with FAISS
- **Vector Database**: Utilized Facebook AI Similarity Search (FAISS) for fast, scalable similarity search
- **Cosine Similarity**: Implemented inner product search on normalized vectors for semantic matching
- **Ranking**: Sorted results by relevance score to prioritize the most relevant assessments

### 4. Query Processing and Constraint Extraction
- **Duration Constraints**: Used regular expressions to identify time limits mentioned in queries
- **Test Type Detection**: Implemented detection of required assessment types (Cognitive, Technical, etc.)
- **Post-filtering**: Applied constraints after semantic search to prioritize matching by meaning

### 5. Dual Interface Implementation
- **Web UI**: Created an intuitive Streamlit interface with different input methods
- **API Endpoint**: Developed a FastAPI service for programmatic access with clear documentation
- **Presentation**: Designed a clean output format showing all required assessment attributes

### 6. Performance Optimization
- **Caching**: Implemented data caching to avoid repeated scraping
- **Efficient Indexing**: Used FAISS's optimized vector indexing for fast query processing
- **Resource Management**: Minimized memory usage through efficient data structures

## Tools and Libraries Used
- **Web Framework**: Streamlit (UI), FastAPI (API)
- **Data Processing**: Pandas, NumPy
- **Web Scraping**: BeautifulSoup4, Requests
- **ML/NLP**: SentenceBERT, FAISS
- **Hosting**: Compatible with Heroku, Google Cloud Run, or AWS Elastic Beanstalk

## Evaluation Strategy
- Implemented logic to compute Mean Recall@3 and MAP@3 metrics
- Validated against the provided sample queries
- Used trace logging to provide insight into the recommendation process

This approach leverages modern NLP techniques to create a semantic understanding of both assessments and queries, providing more intelligent recommendations than traditional keyword-based search systems.