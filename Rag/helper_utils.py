#import dependencies
from PyPDF2 import PdfReader
import requests
from openai import OpenAI
import itertools
import chromadb
import matplotlib.pyplot as plt 
import os 
from dotenv import load_dotenv

load_dotenv()
def load_file(filepath=None):
    """function loads file if filepath is found else  downloads it 

    Args: file_path of document to load 
    
    
    Returns: pages_and_content = {'number' : [],
                         'text' : [],
                          'word_count' : [],
                           'sentence_count' : [],
                           'token_count' : []}
    
    
    """
    pages_and_content = {'number' : [],
                         'text' : [],
                          'word_count' : [],
                           'sentence_count' : [],
                           'token_count' : []}
    
    if os.path.isfile(filepath):
        print('File found, loading...')
        reader = PdfReader(filepath)
        for i, page in enumerate(reader.pages):
            # page_text = clean_text(page.extract_text())  # Uncomment and use if you have a clean_text function
            page_text = str(page.extract_text())
            #page_text = shallow_clean(page_text)
            word_count = len(page_text.split(' '))
            sentence_count = len(page_text.split('.'))
            token_count = len(page_text)/4
            
            pages_and_content['number'].append(i)
            pages_and_content['text'].append(str(page_text))
            pages_and_content['sentence_count'].append(sentence_count)
            pages_and_content['token_count'].append(token_count)
            pages_and_content['word_count'].append(word_count)
        return pages_and_content
    
    else: 
        print('File not available, downloading...')
        response = requests.get(filepath,timeout=30)
        if response.status_code == 200:
            print('File downloaded, saving...')
            with open('data.pdf', 'wb') as f:
                f.write(response.content)
                print('Data saved!')
            reader = PdfReader('data.pdf')
            for i, page in enumerate(reader.pages):
                # page_text = clean_text(page.extract_text())  # Uncomment and use if you have a clean_text function
                page_text = str(page.extract_text())
                #page_text = shallow_clean(page_text)
                word_count = len(page_text.split(' '))
                sentence_count = len(page_text.split('.'))
                token_count = len(page_text)/4
                
                pages_and_content['number'].append(i)
                pages_and_content['text'].append(str(page_text))
                pages_and_content['sentence_count'].append(sentence_count)
                pages_and_content['token_count'].append(token_count)
                pages_and_content['word_count'].append(word_count)

            return pages_and_content 
        else:
            print('Failed to download file')
            return None



def get_openai_response(user_content, model, temperature, client, system_content=None):
    """
    Sends a request to the OpenAI API to generate a response based on the provided user content and context.

    Args:
        user_content (str): The content of the user's query.
        model (str): The name of the OpenAI model to use.
        temperature (float): The temperature parameter 
        controls the randomness of the response.
        client (OpenAI Client): The OpenAI API client to use.
        system_content (str, optional): The system message content to provide context for the response. Default is None.

    Returns:
        dict: The API response containing the generated message.
    """
    messages = [{'role': 'system', 'content': system_content},
                {'role': 'user', 'content': user_content}]
    
    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=temperature)
    
    return response


def create_doc_chunks(text_splitter,page_list):
    """ Creates documents chunks fom list of documents
    
    Args:
        text_spliter - > RecurssiveCharacterTextSpliter from Langchain
        page_list - > pages as a list
        
    Returns:
            document chunks as a list 
    """
    doc_chunks = []

    for page in page_list:
        chunks = text_splitter.split_text(page)
        doc_chunks.append(chunks)
    doc_chunks = list(itertools.chain.from_iterable(doc_chunks))
    return doc_chunks

def token_split(Token_Splitter,doc_chunks):
    """spilts documents into appropriate tokens using using specified chunk size"""
    token_splits =[]
    for text in doc_chunks:
        token_splits += Token_Splitter.split_text(text)
        return token_splits
    
def create_collection(embedding_fcn,database_path,collection_name,token_splits):
    """
    Creates a new collection in a ChromaDB database, adding documents and their corresponding IDs.

    This function connects to a ChromaDB database using a specified path and either retrieves an 
    existing collection or creates a new one. It then adds the provided documents (token_splits) 
    to the collection, using a specified embedding function to generate embeddings.

    Args:
        embedding_fcn (callable): A function used to generate embeddings for the documents.
        database_path (str): The path to the ChromaDB database where the collection is stored.
        collection_name (str): The name of the collection to create or retrieve.
        token_splits (list of str): A list of documents (e.g., text tokens) to be added to the collection.

    Returns:
        chromadb.Collection: The collection object containing the added documents and their embeddings.

    Raises:
        ValueError: If the token_splits list is empty.
        RuntimeError: If there is an issue connecting to the database or adding documents to the collection.
     """

    try:
        if not token_splits:
            raise ValueError('token splits list empty')
        chroma_client = chromadb.PersistentClient(path=database_path)
        collection = chroma_client.get_or_create_collection(name=collection_name,embedding_function=embedding_fcn)
        #addding ids and documents to collection 
        ids = [str(i) for i in range(len(token_splits))]
        collection.add(ids = ids,documents= token_splits)
        return collection
    except Exception as e:
        raise RuntimeError(f'{e}: Could not create collection')
    


#example similarity search
def query_collection(query:list,collection,docs_number):
    """ Funtions Queries a Chroma database and Returns top5 similar documents for every query passed"""
    results = collection.query(query_texts=query,n_results=docs_number,include=['embeddings','documents'])
    return results


def embed_query(query:list,embedding_fcn):
    """" Converts text into embeddings"""
    embedding = embedding_fcn.encode(query)
    return embedding


def supplement_queries(query,client,num_queries):
    """
    Args: qwuery - takes a question 

    Returns - a set of 5 extra questions to support your query
    """
    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to {num_queries} related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
                """
    messages = [{'role':'system',
                 'content': prompt},
                 {'role':'user',
                 'content': query}]
    response = client.chat.completions.create(messages= messages,
                                              model='gpt-3.5-turbo',
                                              temperature = 0.5)
    answer = response.choices[0].message.content
    return answer

def join_queries(original_query, new_queries:list):
    """"  """
    new_queries = new_queries.insert(0,original_query)
    return new_queries





def project_embeddings(embeddings, umap_transform):
    """
    Projects the given embeddings using the provided UMAP transformer.

    Args:
    embeddings (numpy.ndarray): The embeddings to project.
    umap_transform (umap.UMAP): The trained UMAP transformer.

    Returns:
    numpy.ndarray: The projected embeddings.
    """
    projected_embeddings = umap_transform.transform(embeddings)
    return projected_embeddings


#remove duplicate documents
def get_unique_documents(generated_documents:list):
    """ gets unkique documents using sets"""
    unique_documents = set()
    for answers in generated_documents:
        for answer in answers:
            unique_documents.add(answer)
    unique_documents = list(unique_documents)
    return unique_documents

def augment_query_results(joint_query, documents, client, model, temperature=0.5):
    """
    Augments the query results by generating a response from an LLM using the provided documents as context.

    Args:
        joint_query (str): The user's query that needs to be answered.
        documents (list): A list of documents to be used as context for the LLM.
        client (OpenAI Client): The OpenAI API client to be used for generating the response.
        model (str): The name of the model to be used for generating the response.
        temperature (float): The temperature parameter controls the randomness of the response. Default is 0.5.

    Returns:
        str: The generated answer from the LLM.

    Raises:
        ValueError: If the `joint_query` or `documents` are empty.
        RuntimeError: If there is an issue with generating the response from the LLM.
    """
    try:
        if not joint_query:
            raise ValueError("The query cannot be empty.")
        if not documents:
            raise ValueError("The documents list cannot be empty.")

        # Generate unique context from the documents
        context = get_unique_documents(documents)

        # Create system message
        system_content = f"You are an expert financial analyzer who provides responses to '{joint_query}' using the following context as facts from a document search: {context}"

        # Generate the response from the LLM
        response = get_openai_response(user_content=joint_query, 
                                       model=model, 
                                       temperature=temperature, 
                                       client=client, 
                                       system_content=system_content)
        answer = response.choices[0].message.content

        return answer

    except ValueError as ve:
        raise ve
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while generating the response: {e}")

def plot_projections(projected_dataset_embeddings,
                     project_single_query,
                     project_single_query_results,
                     project_augmented_queries,
                     project_aug_query_results_embeddings):
    """" plots all queries both single and multiple queries 
         and their respective search results in the embedding space
         
         Args:       projected_dataset_embeddings,
                     project_single_query,
                     project_single_query_results,
                     project_augmented_queries,
                     project_aug_query_results_embeddings

        Returns: A visualization of  both single and multiple queries 
         and their respective search results in the embedding space
               """
    plt.figure()

    #plot all embddings in dataset
    plt.scatter(
        projected_dataset_embeddings[:, 0],
        projected_dataset_embeddings[:, 1],
        s=10,
        color="gray",
    )
    #plot location of original query
    plt.scatter(project_single_query[:,0],project_single_query[:,1],color = 'blue',label = 'original query')

    plt.scatter(project_single_query_results[:,0],project_single_query_results[:,1],color = 'black',label = 'original query results')

    #plot location of augmented queries
    plt.scatter(project_augmented_queries[:,0],project_augmented_queries[:,1],color = 'red',label = 'augmented queries')

    #plot result document location of joint query results 
    plt.scatter(project_aug_query_results_embeddings[:,0],project_aug_query_results_embeddings[:,1],color = 'green',label = 'augmented query results')

    #plot result location of joint query


    plt.title('Comparison Of Similarity Search for Naive RAG and Expansion Query RAG in Database')
    plt.legend()


 

