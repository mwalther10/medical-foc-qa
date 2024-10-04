from typing import Any
from dotenv import dotenv_values
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

class QdrantOpenAIRetriever():
    
    def __init__(self, collection_name, search_type='similarity_score_threshold', k=10, path_to_env=".env", threshold=0.5, local=False):
        # get environment variables
        ENV = dotenv_values(path_to_env)
        # create qdrant client
        if(local):
            client = QdrantClient(url='localhost', port=6333)
        else:
            client = QdrantClient(url=ENV['QDRANT_URL'])
        self.k = k
        self.search_type = search_type
        self.vectorstore = Qdrant(client, collection_name, OpenAIEmbeddings(api_key=ENV['OPENAI_API_KEY']))
        self.retriever = self.vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": k,"score_threshold": threshold})
    
    def retrieve(self, query: str):
        return self.retriever.invoke(input=query)
    
    def retrieve_unique(self, query: str):
        # run retrieval until k unique results are found
        assert self.search_type == "similarity_score_threshold"
        unique_results = []
        # retrieve top k results
        results = self.vectorstore.similarity_search_with_score(query, 20)
        unique_results = self.remove_duplicates(results)
        return unique_results[:10]
    
    def get_qdrant_id(result):
        return result[0].metadata['_id'] if result[0] is not None else result.metadata['_id']
    
    def get_similarity_score(result):
        return result[1] if result[1] is not None else None
    
    def get_page_content(self, result):
        return result[0].page_content if result[0] is not None else result.page_content
    
    def remove_duplicates(self, results):
        # remove duplicates from results
        unique_results = []
        [unique_results.append(res_A) for res_A in results if self.get_page_content(res_A) not in [self.get_page_content(res_B) for res_B in unique_results]]
        return unique_results
    
    def check_near_duplicates(self, result, unique_results):
        # check if two results are near duplicates
        for unique_result in unique_results:
            tmp = self.vectorstore.similarity_search_with_score(result[0].page_content, k=10)
        return False

    