import subprocess
import json
import os

def run_prolog_query(prolog_file, query):
    """
    Runs a Prolog query using SWI-Prolog and captures the output.
    """
    cmd = ['swipl', '-s', prolog_file, '-g', query, '-t', 'halt']
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def generate_queries(facts):
    """
    Generates sample queries for the given Prolog facts.
    """
    queries = []
    for fact in facts:
        predicate = fact.split('(')[0]
        queries.append(f"{predicate}(X).")
    return queries

def read_prolog_file(file_path):
    """
    Reads a Prolog file and extracts facts.
    """
    with open(file_path, 'r') as file:
        facts = [line.strip() for line in file if line.strip() and not line.startswith('%')]
    return facts

def save_training_data(training_data, output_file):
    """
    Saves the generated training data to a JSON file.
    """
    with open(output_file, 'w') as file:
        json.dump(training_data, file, indent=4)

def main(prolog_file, output_file):
    """
    Main function to generate training data.
    """
    facts = read_prolog_file(prolog_file)
    queries = generate_queries(facts)
    training_data = []

    for query in queries:
        output = run_prolog_query(prolog_file, query)
        training_data.append({
            'prolog_database': open(prolog_file).read(),
            'query': query,
            'output': output
        })

    save_training_data(training_data, output_file)

if __name__ == "__main__":
    prolog_file = 'example.pl'  # Path to your Prolog file
    output_file = 'training_data.json'  # Output JSON file for training data

    main(prolog_file, output_file)
