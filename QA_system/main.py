from QA_system.retriever.qdrant_openai import QdrantOpenAIRetriever
from QA_system.generator.openai import Generator
from langchain_openai import ChatOpenAI



llm = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=1)


retriever = QdrantOpenAIRetriever(collection_name='apothekenumschau', search_type='similarity_score_threshold', k=10)

generator = Generator(retriever)

question = "Was ist die Ursache von Diabetes?"

answer = generator.ask(question)

print(answer)

