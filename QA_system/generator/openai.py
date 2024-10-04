from lifeline_experiments.QA_system.generator.base_generator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, retriever, llm, prompt):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
