import pandas as pd
import matplotlib.pyplot as plt

# Function to plot distance vs. time for each map and model
def plot_model_runs(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Check the first few rows of the dataframe to understand its structure
    print(df.head())
    
    # Assuming the CSV contains columns: 'Time', 'Distance', 'Map', 'Model'
    maps = df['Map'].unique()
    # models = df['Model'].unique()
    models = ['MLP_short', 'MLP_deep', 'MOE_thin', 'MOE_wide', 'ODE_Solver']

    # Plot for each map
    for map_name in maps:
        plt.figure(figsize=(10, 6))  # Create a new figure for each map
        map_data = df[df['Map'] == map_name]  # Filter data for the current map
        
        # Plot the data for each model in this map
        for model in models:
            model_data = map_data[map_data['Model'] == model]
            model_data = model_data.sort_values(by='Nodes')
            plt.plot(model_data['Time'], model_data['Distance'], label=f'Model {model}')
        
        # Customize the plot
        plt.title(f"Distance vs Time for Map '{map_name}'")
        plt.xlabel('Time')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True)
        
        # Show the plot
        plt.show()

# Main function to execute the plotting
if __name__ == "__main__":
    # CSV file path (replace with the path to your CSV file)
    csv_file = 'figures/summary.csv'
    
    plot_model_runs(csv_file)