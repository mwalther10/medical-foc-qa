from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt_template = """ Du bist ein hilfreicher Chatbot, der medizinische Fragen wie ein Hausarzt beantwortet. Beantworte die Frage am Ende des Textes immer anhand der folgenden Informationen.
Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt, versuche nicht, eine Antwort zu erfinden. Verwende leichte Sprache.

{context}

Frage: {question}

Hilfreiche Antwort:"""

class BaseGenerator():
    def __init__(self, retriever, llm, prompt=PromptTemplate.from_template(prompt_template)):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        self.rag_chain = (
            {"context": self.retriever.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask_rag(self, question):
        return self.rag_chain.invoke(question)
    
    def ask_llm(self, prompt):
        return self.llm.invoke(prompt)
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
