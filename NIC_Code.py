import openai
import pinecone
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
# Initialize Pinecone
pc = pinecone.Pinecone(os.environ["PINECONE_API_KEY"])
index = pc.Index("nic-classification")

def search_nic_code(objective):
    # Step 1: Use GPT-4 to refine the user input
    prompt = f"Refine the following business objective for better classification:\n\n{objective}"
    refined_objective = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an expert in business classification."
                   "Post understanding the industry, try to give options which are firstly related to that industry only and then move towards giving generalised answers."
                   "Your focus should be to understand the industry while giving out the code and description. It must align."
                   },
                  {"role": "user", "content": prompt}]
    )["choices"][0]["message"]["content"]

    # Step 2: Generate embedding from the refined objective
    response = openai.Embedding.create(input=refined_objective, model="text-embedding-ada-002")
    query_embedding = response["data"][0]["embedding"]

    # Step 3: Search in Pinecone
    results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

    return [(match["id"], match["metadata"]["description"]) for match in results["matches"]]


# Streamlit UI
st.title("NIC Code Finder")
st.write("Enter your objective below to find the most relevant NIC code.")

user_input = st.text_input("Objective:")

if st.button("Search"):
    if user_input:
        matches = search_nic_code(user_input)
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

