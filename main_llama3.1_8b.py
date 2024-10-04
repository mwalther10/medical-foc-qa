from dotenv import dotenv_values
from langchain_community.llms.ollama import Ollama
from base_setup import BaseSetup
import pandas as pd

# main
def main():
    # Load data
    shortest_queries = pd.read_json("datasets/shortest_queries.json")

    # Initialize the BaseSetup class
    ENV = dotenv_values("../../.env")
    llm = Ollama(model='llama3.1:8b', temperature=0)
    base_setup = BaseSetup(shortest_queries, llm, uses_ollama=True)

    # start experiments
    results = base_setup.run_experiment(batch_size=10, save_batch="llama3.1_8b/shortest", start_from_batch=0)
    results.to_csv("results/results_llama3.1_8b_shortest.csv", sep=";")

if __name__ == "__main__":
    main()