# SHL Assessment Recommendation System

This project is an intelligent recommendation system that helps hiring managers find the right SHL assessments based on job descriptions or natural language queries.

## Features

- Takes natural language queries or job descriptions as input
- Recommends relevant SHL assessments based on semantic matching
- Supports constraints like duration limits and test type preferences
- Provides both a web UI and an API endpoint
- Returns assessment details including name, URL, duration, test type, and support information

## Technical Approach

This solution uses a combination of:

1. **Web Scraping**: BeautifulSoup4 to scrape the SHL product catalog
2. **Vector Embeddings**: SentenceBERT to convert assessment descriptions into vector representations
3. **Similarity Search**: FAISS vector database for efficient semantic matching
4. **Constraint Processing**: Regex and NLP techniques to extract time limits and test type preferences
5. **Web Application**: Streamlit for the UI and FastAPI for the API endpoint

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/shl-recommender.git
cd shl-recommender
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application

For the web UI:
```bash
streamlit run app.py
```

For the API server:
```bash
python app.py api
```

### Using the Web UI

1. Navigate to `http://localhost:8501` in your browser
2. Choose your input type (query, text, or URL)
3. Enter your requirements
4. Click "Recommend Assessments"

### Using the API

Send a GET request to `http://localhost:8000/api/recommend` with your query:

```
GET /api/recommend?query=Looking+to+hire+mid-level+Python+developers
```

## Evaluation Metrics

The system's accuracy can be evaluated using:

- **Mean Recall@K**: The proportion of relevant assessments retrieved in the top K recommendations
- **MAP@K**: Mean Average Precision, which evaluates both relevance and ranking order

## Sample Queries

- "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
- "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes."
- "Here is a JD text, can you recommend some assessment that can help me screen applications. Time limit is less than 30 minutes."
- "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins."

## Limitations and Future Improvements

- The current implementation uses sample data when web scraping fails
- Could be enhanced with additional filters and sorting options
- Performance could be improved with more advanced embedding models
- User feedback could be incorporated to improve recommendations over time

## License

This project is licensed under the MIT License - see the LICENSE file for details.