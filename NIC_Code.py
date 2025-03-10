import asyncio
from concurrent.futures import ThreadPoolExecutor
import openai
import pinecone
import streamlit as st
import os
from dotenv import load_dotenv
import time
from functools import lru_cache
import re
import numpy as np
import requests

# Load environment variables
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Add Microsoft OpenAI fallback credentials
ms_openai_key = os.environ.get("MS_OPENAI_API_KEY")
ms_openai_endpoint = os.environ.get("MS_OPENAI_ENDPOINT")

pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("nic-new-all-data")

class NICFinder:
    def __init__(self):
        self.cache = {}  
        self.cache_hits = 0
        self.total_queries = 0
        self.gpt_model = "gpt-3.5-turbo"  # Faster than gpt-4
        self.embedding_model = "text-embedding-ada-002"
        self.use_ms_openai = False  # Flag to indicate if we should use Microsoft OpenAI
        
    @lru_cache(maxsize=1000)
    def _cache_key(self, objective):
        """Generate cache key for the objective"""
        return f"gpt4:{objective[:100]}"


    async def fetch_embedding(self, text):
        """Fetch embedding with optimized parameters"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            if self.use_ms_openai:
                return await loop.run_in_executor(
                    pool, self._ms_embedding_task,
                    text
                )
            else:
                return await loop.run_in_executor(
                    pool, self._embedding_task,
                    text, self.embedding_model, {"batch_size": 16}
                )

    async def search_pinecone(self, embedding):
        """Search Pinecone with optimized parameters"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return await loop.run_in_executor(
                pool, self._pinecone_task,
                embedding["data"][0]["embedding"],
                20,  # reduced from 30 to 20 for faster results
                True,  # include_metadata
                None,  # filter
                False  # include_values
            )
            
    async def rapid_rerank(self, objective, results, refined_objective):
        """Quickly re-rank search results based on keyword matching and context"""
        matches = results["matches"]
        objective_terms = set(self._extract_keywords(objective.lower() + " " + refined_objective.lower()))
        
        if not objective_terms:  # Fallback if no significant terms extracted
            return [(match["id"], match["metadata"]["description"]) for match in matches[:5]]
        
        # Prepare data for re-ranking
        rerank_data = []
        
        # Get embeddings for objective and refined objective in batch
        objective_combined = objective + " " + refined_objective
        embedding_response = await self.fetch_embedding(objective_combined)
        objective_embedding = embedding_response["data"][0]["embedding"]
        
        # Process all results at once
        for match in matches:
            description = match["metadata"]["description"]
            description_terms = set(self._extract_keywords(description.lower()))
            
            # Calculate simple term overlap (keyword matching)
            common_terms = objective_terms.intersection(description_terms)
            term_score = len(common_terms) / max(1, len(objective_terms))
            
            # Use vector score as a strong signal
            vector_score = match["score"]  # Original similarity score (0-1)
            
            # Simple weighted score - no GPT calls (much faster)
            # Give slightly more weight to vector score for efficiency
            combined_score = (term_score * 0.4) + (vector_score * 0.6)
            
            rerank_data.append({
                "id": match["id"],
                "description": description,
                "original_score": vector_score,
                "term_score": term_score,
                "combined_score": combined_score
            })
        
        # Sort by combined score
        reranked_results = sorted(rerank_data, key=lambda x: x["combined_score"], reverse=True)
        
        # Return top 10 results
        top_results = reranked_results[:5]
        return [(item["id"], item["description"]) for item in top_results]
    
    def _extract_keywords(self, text):
        """Extract meaningful keywords from text"""
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'of', 'in', 'on', 'for', 'to', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text)
        return [word for word in words if word not in stop_words and len(word) > 2]

    async def parallel_search(self, objective):
        """Main search function with caching and parallel processing"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._cache_key(objective)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.total_queries += 1
        
        # Step 1 & 2: Get refined objective and generate embedding in parallel
        loop = asyncio.get_running_loop()
        
        # GPT task with faster model
        prompt = f"Refine the following business objective for better classification:\n\n{objective}"
        
        try:
            print(f"Using OpenAI model: {self.gpt_model}")
            with ThreadPoolExecutor(max_workers=1) as thread_pool:
                gpt_task = loop.run_in_executor(
                    thread_pool, self._gpt_task,
                    prompt
                )
            
            # Embedding task (start early)
            with ThreadPoolExecutor(max_workers=1) as thread_pool:
                embedding_task = loop.run_in_executor(
                    thread_pool, 
                    lambda: openai.Embedding.create(
                        input=objective,
                        model=self.embedding_model,
                        batch_size=16
                    )
                )
            
            # Wait for both tasks to complete
            refined_objective, embedding_response = await asyncio.gather(gpt_task, embedding_task)
        except (openai.error.AuthenticationError, openai.error.APIError, Exception) as e:
            print(f"OpenAI error: {e}")
            print("Switching to Microsoft OpenAI...")
            self.use_ms_openai = True
            
            # Retry with Microsoft OpenAI
            print("Using Microsoft OpenAI model: gpt-35-turbo")
            with ThreadPoolExecutor(max_workers=1) as thread_pool:
                gpt_task = loop.run_in_executor(
                    thread_pool, self._ms_gpt_task,
                    prompt
                )
            
            with ThreadPoolExecutor(max_workers=1) as thread_pool:
                embedding_task = loop.run_in_executor(
                    thread_pool, self._ms_embedding_task,
                    objective
                )
            
            # Wait for both tasks to complete
            refined_objective, embedding_response = await asyncio.gather(gpt_task, embedding_task)

        
        # Step 3: Search Pinecone
        raw_results = await self.search_pinecone(embedding_response)
        
        # Step 4: Fast contextual re-ranking (no additional API calls)
        reranked_results = await self.rapid_rerank(objective, raw_results, refined_objective)
        
        # Cache results
        self.cache[cache_key] = reranked_results
        
        print(f"Query time: {time.time() - start_time:.2f}s")
        print(f"Cache hit rate: {self.cache_hits/self.total_queries:.2%}")
        
        return reranked_results

    def _gpt_task(self, prompt):
        """Helper function for GPT-4 queries using OpenAI"""
        return openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[{
                "role": "system",
                "content":  f"""You are an expert in business classification, specializing in NIC codes. 

                        Your task is to refine business objectives for precise classification.

                        ### **Classification Guidelines:**
                        1. **Identify the Primary Industry**  
                        - Classify the business based on its **core activity** (e.g., real estate, manufacturing, healthcare).  
                        - **DO NOT** suggest NIC codes that belong to adjacent or indirectly related industries.  

                        2. **Strict Industry Matching**  
                        - If the objective is **real estate**, only suggest NIC codes from the **68xxx** series.  
                        - **DO NOT** suggest general brokerage (74xxx) or insurance (66xxx), even if they seem applicable.  

                        3. **Human-Focused Manufacturing**  
                        - If asked for **food manufacturing**, suggest only **human food processing** (e.g., bread, dairy, snacks, packaged foods).  
                        - **DO NOT** suggest cattle feed, poultry feed, or pet food, as they belong to a separate category. Only highlight NIC codes relevant to human food processing.  
                        - Example:  
                            - ✅ "Manufacture of bakery products" → Suggest **NIC code 1071**  
                            - ❌ "Manufacture of food" → **DO NOT** suggest animal feed codes like 1080 
                            - ✅ "Manufacture of food" → Suggest **NIC code 1071** and other relevant human foods but **NEVER** suggest 10801 and 10802!

                        4, Until specifically stated, do not consider food as animal food. Consider "food" as human food while a question regarding manufacture of food is asked.

                        5. **Avoid Peripheral Activities**  
                        - Ignore secondary or auxiliary activities. Focus only on the business's **primary function**.  

                        ### **STRICT EXCLUSION:**  
                        🚫 **NEVER** suggest:  
                        - Animal feed, poultry feed, or any non-human food products when asked about food manufacturing.  
                        - NIC codes from unrelated industries, even if there is a minor overlap.  

                        Follow these rules **precisely** and ensure that suggestions are accurate.


                    """""
            },
            {
                "role": "user",
                "content": prompt
            }]
        )["choices"][0]["message"]["content"]

    def _ms_gpt_task(self, prompt):
        """Helper function for GPT queries using Microsoft Azure OpenAI"""
        if not ms_openai_key or not ms_openai_endpoint:
            raise ValueError("Microsoft OpenAI credentials not found in environment variables")
        
        system_content = """You are an expert in business classification, specializing in NIC codes. 

                Your task is to refine business objectives for precise classification.

                ### **Classification Guidelines:**
                1. **Identify the Primary Industry**  
                - Classify the business based on its **core activity** (e.g., real estate, manufacturing, healthcare).  
                - **DO NOT** suggest NIC codes that belong to adjacent or indirectly related industries.  

                2. **Strict Industry Matching**  
                - If the objective is **real estate**, only suggest NIC codes from the **68xxx** series.  
                - **DO NOT** suggest general brokerage (74xxx) or insurance (66xxx), even if they seem applicable.  

                3. **Human-Focused Manufacturing**  
                - If asked for **food manufacturing**, suggest only **human food processing** (e.g., bread, dairy, snacks, packaged foods).  
                - **DO NOT** suggest cattle feed, poultry feed, or pet food, as they belong to a separate category. Only highlight NIC codes relevant to human food processing.  
                - Example:  
                    - ✅ "Manufacture of bakery products" → Suggest **NIC code 1071**  
                    - ❌ "Manufacture of food" → **DO NOT** suggest animal feed codes like 1080 
                    - ✅ "Manufacture of food" → Suggest **NIC code 1071** and other relevant human foods but **NEVER** suggest 10801 and 10802!

                4, Until specifically stated, do not consider food as animal food. Consider "food" as human food while a question regarding manufacture of food is asked.

                5. **Avoid Peripheral Activities**  
                - Ignore secondary or auxiliary activities. Focus only on the business's **primary function**.  

                ### **STRICT EXCLUSION:**  
                🚫 **NEVER** suggest:  
                - Animal feed, poultry feed, or any non-human food products when asked about food manufacturing.  
                - NIC codes from unrelated industries, even if there is a minor overlap.  

                Follow these rules **precisely** and ensure that suggestions are accurate."""
        
        headers = {
            "Content-Type": "application/json",
            "api-key": ms_openai_key
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            "model": "gpt-35-turbo",  # Use Azure's equivalent model
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        deployment_name = "gpt-35-turbo"  # Replace with your Azure OpenAI deployment name
        endpoint = f"{ms_openai_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version=2024-10-21"
        
        response = requests.post(endpoint, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Microsoft OpenAI API error: {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]

    def _embedding_task(self, text, model, params=None):
        """Helper function for embedding generation using OpenAI"""
        return openai.Embedding.create(input=text, model=model, **(params or {}))

    # def _ms_embedding_task(self, text):
    #     """Helper function for embedding generation using Microsoft Azure OpenAI"""
    #     if not ms_openai_key or not ms_openai_endpoint:
    #         raise ValueError("Microsoft OpenAI credentials not found in environment variables")
        
    #     headers = {
    #         "Content-Type": "application/json",
    #         "api-key": ms_openai_key
    #     }
        
    #     payload = {
    #         "input": text,
    #         "dimensions": 1536  # Standard for text-embedding-ada-002
    #     }
        
    #     deployment_name = "text-embedding-ada-002"  # Replace with your Azure OpenAI deployment name
    #     endpoint = f"{ms_openai_endpoint}/openai/deployments/{deployment_name}/embeddings?api-version=2023-05-15"
        
    #     response = requests.post(endpoint, headers=headers, json=payload)
        
    #     if response.status_code != 200:
    #         raise Exception(f"Microsoft OpenAI API error: {response.text}")
        
    #     # Format the response to match OpenAI's API response structure
    #     result = response.json()
    #     formatted_response = {
    #         "data": [{"embedding": result["data"][0]["embedding"]}]
    #     }
        
    #     return formatted_response

    def _ms_embedding_task(self, text):
        """Helper function for embedding generation using Microsoft Azure OpenAI"""
        if not ms_openai_key or not ms_openai_endpoint:
            raise ValueError("Microsoft OpenAI credentials not found in environment variables")
        
        headers = {
            "Content-Type": "application/json",
            "api-key": ms_openai_key
        }
        
        # Remove the dimensions parameter that's causing the error
        payload = {
            "input": text
        }
        
        deployment_name = "text-embedding-ada-002"  # Replace with your Azure OpenAI deployment name
        endpoint = f"{ms_openai_endpoint}/openai/deployments/{deployment_name}/embeddings?api-version=2023-05-15"
        
        response = requests.post(endpoint, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Microsoft OpenAI API error: {response.text}")
        
        # Format the response to match OpenAI's API response structure
        result = response.json()
        formatted_response = {
            "data": [{"embedding": result["data"][0]["embedding"]}]
        }
        
        return formatted_response

    def _pinecone_task(self, embedding, top_k, include_metadata, filter=None, include_values=False):
        """Helper function for Pinecone search"""
        return index.query(vector=embedding, top_k=top_k, include_metadata=include_metadata,
                         filter=filter, include_values=include_values)

# Streamlit UI
finder = NICFinder()

st.title("NIC Code Finder")
st.write("Enter your objective below to find the most relevant NIC code.")

user_input = st.text_input("Objective:")

if st.button("Search"):
    if user_input:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Display which service is being used in the Streamlit UI
        st.info("Processing your request...")
        
        matches = loop.run_until_complete(finder.parallel_search(user_input))
        loop.close()
        
        # Show which service was used
        if finder.use_ms_openai:
            st.success("Used Microsoft OpenAI service (Fallback mode)")
        else:
            st.success(f"Used OpenAI service (Model: {finder.gpt_model})")
        
        if matches:
            st.write("### Top Matches:")
            for code, description in matches:
                st.write(f"**NIC Code:** {code}")
                st.write(f"**Description:** {description}")
                st.write("---")
        else:
            st.write("No matches found. Describe the objective better.")
    else:
        st.write("Please enter an objective to search.")
