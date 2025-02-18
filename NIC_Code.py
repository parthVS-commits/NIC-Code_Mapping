
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import openai
import pinecone
import streamlit as st
import os
from dotenv import load_dotenv
import time
from functools import lru_cache

# Load environment variables
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("nic-classification")

class NICFinder:
    def __init__(self):
        self.cache = {}  
        self.cache_hits = 0
        self.total_queries = 0
        self.gpt_model = "gpt-3.5-turbo"  # Faster than gpt-4
        self.embedding_model = "text-embedding-ada-002"
        
    @lru_cache(maxsize=1000)
    def _cache_key(self, objective):
        """Generate cache key for the objective"""
        return f"gpt4:{objective[:100]}"

    async def fetch_embedding(self, text):
        """Fetch embedding with optimized parameters"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
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
                3,  # top_k
                True,  # include_metadata
                None,  # filter
                False  # include_values
            )

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
        tasks = []
        
        # GPT task with faster model
        prompt = f"Refine the following business objective for better classification:\n\n{objective}"
        with ProcessPoolExecutor(max_workers=1) as process_pool:
            gpt_task = loop.run_in_executor(
                process_pool, self._gpt_task,
                prompt
            )
            tasks.append(gpt_task)
        
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
            tasks.append(embedding_task)
        
        # Wait for both tasks to complete
        refined_objective = await gpt_task
        embedding_response = await embedding_task
        
        # Step 3: Search Pinecone
        results = await self.search_pinecone(embedding_response)
        
        # Cache results
        matches = [(match["id"], match["metadata"]["description"]) 
                  for match in results["matches"]]
        self.cache[cache_key] = matches
        
        print(f"Query time: {time.time() - start_time:.2f}s")
        print(f"Cache hit rate: {self.cache_hits/self.total_queries:.2%}")
        
        return matches

    def _gpt_task(self, prompt):
        """Helper function for GPT-4 queries"""
        return openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[{
                "role": "system",
                "content": "You are an expert in business classification."
                          "Post understanding the industry, try to give options which are firstly related to that industry only and then move towards giving generalised answers."
                          "Your focus should be to understand the industry while giving out the code and description. It must align."
            },
            {
                "role": "user",
                "content": prompt
            }]
        )["choices"][0]["message"]["content"]

    def _embedding_task(self, text, model, params=None):
        """Helper function for embedding generation"""
        return openai.Embedding.create(input=text, model=model, **(params or {}))

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
        matches = loop.run_until_complete(finder.parallel_search(user_input))
        loop.close()
        
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
