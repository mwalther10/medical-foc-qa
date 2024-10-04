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
    llm = Ollama(model='gemma2', temperature=0)
    base_setup = BaseSetup(shortest_queries, llm, uses_ollama=True)

    # start experiments
    results = base_setup.run_experiment(batch_size=10, save_batch="gemma2/shortest", start_from_batch=0)
    results.to_csv("results/results_gemma2_shortest.csv", sep=";")

if __name__ == "__main__":
    main()