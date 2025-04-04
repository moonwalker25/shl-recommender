# Main application file: app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import time

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight, efficient model for embeddings
SHL_CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
MAX_RECOMMENDATIONS = 10

class SHLAssessmentRecommender:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.assessment_data = []
        self.embeddings = None
        self.index = None
        self.load_or_scrape_data()
        self.build_index()
    
    def load_or_scrape_data(self):
        """Load existing data or scrape from SHL website"""
        try:
            # Try to load cached data first
            self.assessment_data = pd.read_csv("shl_assessments.csv").to_dict('records')
            print(f"Loaded {len(self.assessment_data)} assessments from cache")
        except FileNotFoundError:
            # If no cached data, scrape the website
            print("Scraping SHL website for assessment data...")
            self.assessment_data = self.scrape_shl_catalog()
            # Save to cache
            pd.DataFrame(self.assessment_data).to_csv("shl_assessments.csv", index=False)
    
    def scrape_shl_catalog(self) -> List[Dict[str, Any]]:
        """Scrape the SHL product catalog and extract assessment details"""
        try:
            response = requests.get(SHL_CATALOG_URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            assessments = []
            
            # Find all product cards/sections
            product_sections = soup.find_all('div', class_='product-card')  # Adjust selector based on actual HTML
            
            for product in product_sections:
                try:
                    # Extract relevant information (adjust selectors based on actual HTML)
                    name = product.find('h3').text.strip() if product.find('h3') else "Unknown"
                    url = product.find('a')['href'] if product.find('a') else SHL_CATALOG_URL
                    description = product.find('div', class_='description').text.strip() if product.find('div', class_='description') else ""
                    
                    # Extract features like duration, remote testing, adaptive support
                    features_section = product.find('div', class_='features') if product.find('div', class_='features') else product
                    
                    # Example extraction logic (adjust based on actual HTML)
                    duration_match = re.search(r'(\d+)\s*minutes', features_section.text) if features_section else None
                    duration = duration_match.group(1) if duration_match else "Varies"
                    
                    remote_testing = "Yes" if "remote" in features_section.text.lower() else "No" if features_section else "Unknown"
                    adaptive_support = "Yes" if "adaptive" in features_section.text.lower() or "irt" in features_section.text.lower() else "No" if features_section else "Unknown"
                    
                    # Determine test type
                    test_type = self.determine_test_type(name, description)
                    
                    assessment = {
                        "name": name,
                        "url": url if url.startswith("http") else f"https://www.shl.com{url}",
                        "description": description,
                        "duration": duration,
                        "remote_testing": remote_testing,
                        "adaptive_support": adaptive_support,
                        "test_type": test_type,
                        "search_text": f"{name} {description} {test_type}"
                    }
                    
                    assessments.append(assessment)
                except Exception as e:
                    print(f"Error processing product: {e}")
            
            # Fallback to sample data if scraping fails or returns empty
            if not assessments:
                assessments = self.get_sample_data()
                
            return assessments
        except Exception as e:
            print(f"Error scraping SHL catalog: {e}")
            # Return sample data if scraping fails
            return self.get_sample_data()
    
    def get_sample_data(self) -> List[Dict[str, Any]]:
        """Generate sample assessment data if scraping fails"""
        return [
            {
                "name": "Java Developer Assessment",
                "url": "https://www.shl.com/solutions/products/java-developer-assessment/",
                "description": "Evaluates Java programming skills, problem-solving abilities, and software development practices.",
                "duration": "40",
                "remote_testing": "Yes",
                "adaptive_support": "Yes",
                "test_type": "Technical",
                "search_text": "Java Developer Assessment Evaluates Java programming skills, problem-solving abilities, and software development practices. Technical Java programming coding software development collaboration"
            },
            {
                "name": "Python Coding Test",
                "url": "https://www.shl.com/solutions/products/python-coding-test/",
                "description": "Assesses proficiency in Python programming language with real-world problems and coding tasks.",
                "duration": "30",
                "remote_testing": "Yes",
                "adaptive_support": "No",
                "test_type": "Technical",
                "search_text": "Python Coding Test Assesses proficiency in Python programming language with real-world problems and coding tasks. Technical Python programming coding algorithms data structures"
            },
            {
                "name": "SQL Database Skills Assessment",
                "url": "https://www.shl.com/solutions/products/sql-database-skills/",
                "description": "Tests SQL query writing, database design understanding, and data manipulation skills.",
                "duration": "35",
                "remote_testing": "Yes",
                "adaptive_support": "No",
                "test_type": "Technical",
                "search_text": "SQL Database Skills Assessment Tests SQL query writing, database design understanding, and data manipulation skills. Technical SQL database queries data manipulation"
            },
            {
                "name": "JavaScript Frontend Developer Test",
                "url": "https://www.shl.com/solutions/products/javascript-developer-test/",
                "description": "Evaluates JavaScript, HTML, CSS skills with emphasis on modern frameworks and web development principles.",
                "duration": "45",
                "remote_testing": "Yes",
                "adaptive_support": "No",
                "test_type": "Technical",
                "search_text": "JavaScript Frontend Developer Test Evaluates JavaScript, HTML, CSS skills with emphasis on modern frameworks and web development principles. Technical JavaScript HTML CSS frameworks web development"
            },
            {
                "name": "Full Stack Developer Assessment Pack",
                "url": "https://www.shl.com/solutions/products/full-stack-developer-pack/",
                "description": "Comprehensive package testing both frontend and backend skills including JavaScript, Python, SQL, and system design.",
                "duration": "60",
                "remote_testing": "Yes",
                "adaptive_support": "Yes",
                "test_type": "Technical",
                "search_text": "Full Stack Developer Assessment Pack Comprehensive package testing both frontend and backend skills including JavaScript, Python, SQL, and system design. Technical Full Stack JavaScript Python SQL system design"
            },
            {
                "name": "Cognitive Ability Assessment",
                "url": "https://www.shl.com/solutions/products/cognitive-ability-assessment/",
                "description": "Measures problem-solving, critical thinking, and learning ability across verbal, numerical, and logical reasoning domains.",
                "duration": "25",
                "remote_testing": "Yes",
                "adaptive_support": "Yes",
                "test_type": "Cognitive",
                "search_text": "Cognitive Ability Assessment Measures problem-solving, critical thinking, and learning ability across verbal, numerical, and logical reasoning domains. Cognitive problem-solving critical thinking reasoning"
            },
            {
                "name": "Personality Insights Assessment",
                "url": "https://www.shl.com/solutions/products/personality-insights/",
                "description": "Evaluates work behaviors, communication styles, and collaboration preferences to determine cultural fit.",
                "duration": "20",
                "remote_testing": "Yes",
                "adaptive_support": "No",
                "test_type": "Personality",
                "search_text": "Personality Insights Assessment Evaluates work behaviors, communication styles, and collaboration preferences to determine cultural fit. Personality behaviors communication collaboration teamwork"
            },
            {
                "name": "Business Analyst Aptitude Test",
                "url": "https://www.shl.com/solutions/products/business-analyst-aptitude/",
                "description": "Assesses data interpretation, requirements gathering, and analytical thinking skills for business analysis roles.",
                "duration": "30",
                "remote_testing": "Yes",
                "adaptive_support": "Yes",
                "test_type": "Technical",
                "search_text": "Business Analyst Aptitude Test Assesses data interpretation, requirements gathering, and analytical thinking skills for business analysis roles. Technical Business Analyst data interpretation requirements analysis"
            },
            {
                "name": "Data Science Skills Assessment",
                "url": "https://www.shl.com/solutions/products/data-science-skills/",
                "description": "Tests statistical analysis, machine learning concepts, and data visualization skills for data science positions.",
                "duration": "50",
                "remote_testing": "Yes",
                "adaptive_support": "Yes",
                "test_type": "Technical",
                "search_text": "Data Science Skills Assessment Tests statistical analysis, machine learning concepts, and data visualization skills for data science positions. Technical Data Science statistics machine learning data visualization"
            },
            {
                "name": "Leadership Potential Assessment",
                "url": "https://www.shl.com/solutions/products/leadership-potential/",
                "description": "Evaluates decision-making, team management, and strategic thinking capabilities for leadership roles.",
                "duration": "40",
                "remote_testing": "Yes",
                "adaptive_support": "No",
                "test_type": "Behavioral",
                "search_text": "Leadership Potential Assessment Evaluates decision-making, team management, and strategic thinking capabilities for leadership roles. Behavioral Leadership decision-making management strategic thinking"
            },
            {
                "name": "Situational Judgment Test",
                "url": "https://www.shl.com/solutions/products/situational-judgment/",
                "description": "Measures judgment and decision-making in workplace scenarios related to ethics, team dynamics, and problem resolution.",
                "duration": "30",
                "remote_testing": "Yes",
                "adaptive_support": "No",
                "test_type": "Behavioral",
                "search_text": "Situational Judgment Test Measures judgment and decision-making in workplace scenarios related to ethics, team dynamics, and problem resolution. Behavioral judgment decision-making workplace scenarios"
            },
            {
                "name": "Communication Skills Assessment",
                "url": "https://www.shl.com/solutions/products/communication-skills/",
                "description": "Evaluates written and verbal communication abilities, active listening, and presentation skills.",
                "duration": "25",
                "remote_testing": "Yes",
                "adaptive_support": "No",
                "test_type": "Soft Skills",
                "search_text": "Communication Skills Assessment Evaluates written and verbal communication abilities, active listening, and presentation skills. Soft Skills communication written verbal listening presentation collaboration"
            }
        ]
    
    def determine_test_type(self, name: str, description: str) -> str:
        """Determine the test type based on name and description"""
        text = (name + " " + description).lower()
        
        if any(term in text for term in ["java", "python", "sql", "javascript", "coding", "programming"]):
            return "Technical"
        elif any(term in text for term in ["cognitive", "verbal", "numerical", "logical", "reasoning"]):
            return "Cognitive"
        elif any(term in text for term in ["personality", "behavior"]):
            return "Personality"
        elif any(term in text for term in ["leadership", "management", "situational"]):
            return "Behavioral"
        elif any(term in text for term in ["communication", "soft skills"]):
            return "Soft Skills"
        else:
            return "Other"
    
    def build_index(self):
        """Build FAISS index for efficient similarity search"""
        # Prepare search texts for embedding
        search_texts = [item["search_text"] for item in self.assessment_data]
        
        # Generate embeddings
        self.embeddings = self.model.encode(search_texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Build FAISS index
        vector_dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(vector_dimension)  # Inner product for cosine similarity with normalized vectors
        self.index.add(self.embeddings)
    
    def recommend(self, query: str, max_results: int = MAX_RECOMMENDATIONS) -> List[Dict[str, Any]]:
        """Recommend assessments based on query"""
        # Process the query to extract constraints
        duration_constraint = self.extract_duration_constraint(query)
        test_type_constraints = self.extract_test_type_constraints(query)
        
        # Embed the query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar assessments
        scores, indices = self.index.search(query_embedding, len(self.assessment_data))
        
        # Get all results and apply constraints
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:  # Valid index
                assessment = self.assessment_data[idx].copy()
                assessment["relevance_score"] = float(score)
                
                # Apply constraints
                if duration_constraint and int(assessment["duration"]) > duration_constraint:
                    continue
                
                if test_type_constraints and assessment["test_type"] not in test_type_constraints:
                    continue
                
                results.append(assessment)
        
        # Sort by relevance score and limit results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]
    
    def extract_duration_constraint(self, query: str) -> Optional[int]:
        """Extract time/duration constraint from query"""
        # Look for patterns like "under 30 minutes", "less than 45 min", etc.
        patterns = [
            r'under (\d+) min',
            r'less than (\d+) min',
            r'within (\d+) min',
            r'max.*?(\d+) min',
            r'maximum.*?(\d+) min',
            r'not exceed.*?(\d+) min',
            r'completed in (\d+) min'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def extract_test_type_constraints(self, query: str) -> List[str]:
        """Extract test type constraints from query"""
        test_types = []
        
        # Check for specific test types mentioned
        if re.search(r'cognitive|reasoning|aptitude', query.lower()):
            test_types.append("Cognitive")
        
        if re.search(r'technical|coding|programming|developer|java|python|sql|javascript', query.lower()):
            test_types.append("Technical")
        
        if re.search(r'personality|behavior', query.lower()):
            test_types.append("Personality")
        
        if re.search(r'leadership|situational|judgment', query.lower()):
            test_types.append("Behavioral")
        
        if re.search(r'communication|soft skills', query.lower()):
            test_types.append("Soft Skills")
        
        return test_types
    
    def fetch_job_description(self, url: str) -> str:
        """Fetch job description from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Basic extraction of text content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            return f"Error fetching job description: {str(e)}"

# Streamlit UI
def create_ui():
    st.set_page_config(page_title="SHL Assessment Recommender", page_icon="🧪", layout="wide")
    
    st.title("SHL Assessment Recommendation System")
    st.markdown("""
    This application helps hiring managers find the right SHL assessments for their roles.
    Enter a natural language query or paste a job description, and get relevant assessment recommendations.
    """)
    
    # Initialize recommender (with caching)
    @st.cache_resource
    def get_recommender():
        return SHLAssessmentRecommender()
    
    recommender = get_recommender()
    
    # Input options
    input_type = st.radio("Select input type:", ["Natural Language Query", "Job Description Text", "Job Description URL"])
    
    query = ""
    
    if input_type == "Natural Language Query":
        query = st.text_area("Enter your requirements:", 
                            placeholder="Example: Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
                            height=100)
    
    elif input_type == "Job Description Text":
        query = st.text_area("Paste the job description:", 
                            placeholder="Paste the full job description here...",
                            height=200)
    
    elif input_type == "Job Description URL":
        url = st.text_input("Enter job description URL:", 
                          placeholder="https://example.com/job-posting")
        if url and st.button("Fetch Job Description"):
            with st.spinner("Fetching job description..."):
                query = recommender.fetch_job_description(url)
                st.text_area("Fetched job description:", value=query, height=200)
    
    # Process button
    if st.button("Recommend Assessments") and query:
        with st.spinner("Finding relevant assessments..."):
            recommendations = recommender.recommend(query)
            
            if recommendations:
                st.success(f"Found {len(recommendations)} relevant assessments")
                
                # Display recommendations in a table
                df = pd.DataFrame(recommendations)
                display_cols = ["name", "test_type", "duration", "remote_testing", "adaptive_support", "relevance_score"]
                df_display = df[display_cols].copy()
                df_display["relevance_score"] = df_display["relevance_score"].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(df_display, use_container_width=True)
                
                # Display detailed cards for each recommendation
                st.subheader("Detailed Recommendations")
                
                for i, rec in enumerate(recommendations):
                    with st.expander(f"{i+1}. {rec['name']} - {rec['test_type']}"):
                        st.markdown(f"**Name**: [{rec['name']}]({rec['url']})")
                        st.markdown(f"**Test Type**: {rec['test_type']}")
                        st.markdown(f"**Duration**: {rec['duration']} minutes")
                        st.markdown(f"**Remote Testing Support**: {rec['remote_testing']}")
                        st.markdown(f"**Adaptive/IRT Support**: {rec['adaptive_support']}")
                        st.markdown(f"**Description**: {rec['description']}")
                        st.markdown(f"**Relevance Score**: {rec['relevance_score']:.2f}")
            else:
                st.warning("No matching assessments found. Try adjusting your query.")
    
    # API Documentation
    with st.expander("API Documentation"):
        st.markdown("""
        ## API Documentation
        
        The recommendation system also provides an API endpoint for programmatic access.
        
        ### GET Endpoint
        
        ```
        GET /api/recommend?query=your+query+here
        ```
        
        ### Parameters
        
        - `query` (required): Your natural language query or job description
        - `max_results` (optional): Maximum number of results to return (default: 10)
        
        ### Example Response
        
        ```json
        {
            "recommendations": [
                {
                    "name": "Java Developer Assessment",
                    "url": "https://www.shl.com/solutions/products/java-developer-assessment/",
                    "description": "Evaluates Java programming skills...",
                    "duration": "40",
                    "remote_testing": "Yes",
                    "adaptive_support": "Yes",
                    "test_type": "Technical",
                    "relevance_score": 0.87
                },
                ...
            ]
        }
        ```
        """)
    
    # Add footer
    st.markdown("---")
    st.markdown("SHL Assessment Recommendation System - Built with Streamlit, sentence-transformers, and FAISS")

# API Server (using FastAPI)
def create_api():
    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    app = FastAPI(title="SHL Assessment Recommender API")
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize recommender
    recommender = SHLAssessmentRecommender()
    
    @app.get("/api/recommend")
    async def recommend(
        query: str = Query(..., description="Natural language query or job description"),
        max_results: int = Query(10, description="Maximum number of results to return", ge=1, le=20)
    ):
        recommendations = recommender.recommend(query, max_results)
        return {"recommendations": recommendations}
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Main entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run as API server
        create_api()
    else:
        # Run as Streamlit app
        create_ui()
