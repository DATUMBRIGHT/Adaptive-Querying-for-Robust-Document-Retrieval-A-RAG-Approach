import os
import streamlit as st
import umap
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from helper_utils import (
                            load_file,
                            project_embeddings,
                            create_doc_chunks, 
                            token_split, create_collection, 
                            query_collection,
                            embed_query, supplement_queries, 
                            plot_projections, augment_query_results
                            )


load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key = OPENAI_KEY)


# Streamlit app setup
st.title('Financial Report Analyzer with RAG and OpenAI')
st.sidebar.title('Configuration')

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
upload_button = st.button(label= 'Upload document')
# Create a temporary file

if uploaded_file is not None and upload_button:
    # Save the uploaded file to a directory
    file_dir = "./uploads"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        pages_and_content = load_file(file_path)
        st.write("PDF loaded successfully!")
        st.write(pd.DataFrame(pages_and_content))
    except Exception as e:
        st.error(f'File load failure: {e}')

    with st.form(key='query_form'):
        query = st.text_input('What do you want to search in document?', placeholder='Type your query here')
        submit_button = st.form_submit_button(label='Go!')

        if submit_button:
            st.write("Button clicked.....")

           
            try:
                # Text splitting
                st.write('Splitting document...')
                text_splitter = RecursiveCharacterTextSplitter(separators=['.', '\n'])
                docs = pages_and_content['text']
                doc_chunks = create_doc_chunks(text_splitter, docs)

                # Token splitting
                token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
                token_splits = token_split(token_splitter, doc_chunks)

                # Create collection
                db_dir = './db'
                collection_name = os.path.basename(file_path)
                collection_path = os.path.join(db_dir, collection_name + '.db')
                os.makedirs(db_dir, exist_ok=True)

                if not os.path.exists(collection_path):
                    embedding_fcn = SentenceTransformer("all-MiniLM-L6-v2")
                    collection = create_collection(embedding_fcn, db_dir, collection_name, token_splits)
                    st.write('Collection created successfully')

                # Project dataset embeddings
                dataset_embeddings = collection.get(include=['embeddings', 'documents'])['embeddings']
                umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(dataset_embeddings)
                project_dataset_embeddings = project_embeddings(dataset_embeddings, umap_transform)

                # Querying
                single_query_embedding = embed_query([query], embedding_fcn)
                project_single_query = project_embeddings(single_query_embedding, umap_transform)

                # Query results
                results = query_collection([query], collection, 5)
                project_single_query_result_embedding = project_embeddings(results['embeddings'],umap_transform)
                st.write('Query results:', results['documents'])

                # Further code for joint queries, answer generation, and plotting...
                # Joint queries
                joint_queries = supplement_queries(query, client, 5)
                joint_queries.insert(0, query)  # Include the original query
                
                # Display single query and multiple queries
                st.write(f'Your query: {query}')
                st.write(f'Generated queries: {joint_queries[1:]}')
                
                joint_queries_embeddings = embedding_fcn(joint_queries)
                project_joint_queries = project_embeddings(joint_queries_embeddings, umap_transform)
                
                # Joint query results
                docs_number = 5
                joint_queries_results = query_collection(joint_queries, collection, docs_number)
                joint_queries_results_embeddings = joint_queries_results['embeddings']
                project_joint_queries_result_embeddings = project_embeddings(joint_queries_results_embeddings, umap_transform)
                
                st.write('Query results:')
                st.write(results['documents'])
                st.write(f"Multi-query results: {joint_queries_results['documents']}")
                
                # Generate the final answer
                documents = joint_queries_results['documents']
                model = 'gpt-3.5-turbo'
                try:
                    answer = augment_query_results(joint_queries, documents, client, model)
                    st.write(f'Answer: {answer}')
                    
                    # Plot projections
                    st.write('Plotting projections...')
                    st.write(plot_projections(
                        projected_dataset_embeddings=project_dataset_embeddings,
                        project_single_query=project_single_query,
                        project_single_query_results=project_single_query_result_embedding,
                        project_augmented_queries=project_joint_queries,
                        project_aug_query_results_embeddings=project_joint_queries_result_embeddings)
                    )
                except Exception as e:
                        st.error(f'Error: {e}. Could not process the query.')
            except Exception as e :
                st.error(f'{e} couldnt process query')

