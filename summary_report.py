import os
import yaml
import csv

# Function to load YAML data from a file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to extract relevant data from the configuration and results YAML files
def extract_data(model_run_folder):
    # Paths to the YAML files
    config_file = os.path.join(model_run_folder, 'configuration.yaml')
    results_file = os.path.join(model_run_folder, 'results.yaml')
    
    # Load the YAML data
    config_data = load_yaml(config_file)
    results_data = load_yaml(results_file)
    
    # Extract the relevant fields from the configuration and results (modify these as needed)
    config_values = {
        'Model Type': config_data.get('model', None).get('type', None),
        'Hidden Dims': config_data.get('model', None).get('hidden_dims', None),
        'Num Experts': config_data.get('model', None).get('num_experts', None),
        'Top K': config_data.get('model', None).get('top_k', None),
    }
    
    results_values = {
        'Loss': results_data.get('loss', None),
        'XY Distance': results_data.get('xy_diff', None),
        'Theta Error': results_data.get('theta_error', None),
    }
    
    # Combine config values and results values
    return {**config_values, **results_values}

# Function to write the data to a CSV file
def write_to_csv(data, output_csv):
    # Writing to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()  # Write the header
        writer.writerows(data)  # Write the data rows

# Main function to process the folders
def process_experiments(experiment_folder, output_csv):
    # List to hold data for all experiments
    all_data = []
    
    # Iterate over the experiment folders
    for model_run in os.listdir(experiment_folder):
        model_run_path = os.path.join(experiment_folder, model_run)
        
        # Check if it's a directory (valid experiment)
        if os.path.isdir(model_run_path):
            # Extract the data for this experiment
            model_run_data = extract_data(model_run_path)
            all_data.append(model_run_data)
    
    # Write all the collected data to a CSV file
    write_to_csv(all_data, output_csv)
    print(f"Data has been written to {output_csv}")

# Set the runs folder and output CSV file path
runs_folder = 'runs'  # Modify with the path to your runs folder
experiment_folder_name = 'Exp1_12_17_19_54_07'
output_csv = 'experiment_results.csv'  # Output CSV file

experiment_path = f'{runs_folder}/{experiment_folder_name}'
output_path = f'{experiment_path}/{output_csv}'

# Call the function to process experiments
process_experiments(experiment_path, output_path)