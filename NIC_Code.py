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

# Function to search for the closest NIC code based on user objective
def search_nic_code(objective):
    response = openai.Embedding.create(input=objective, model="text-embedding-ada-002")
    query_embedding = response["data"][0]["embedding"]
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
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

