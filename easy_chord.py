#%%
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from mpl_chord_diagram import chord_diagram

# Set global font
plt.rcParams['font.family'] = 'Times New Roman'

# Load CSV file
def load_csv(file_path):
    """
    Load a CSV file where the first column contains feature names.
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the data.
    """
    df = pd.read_csv(file_path)
    return df

# Plot Chord diagram
def plot_chord_diagram(df, save_path=None):
    """
    Generate a Chord diagram based on the correlation matrix of the DataFrame.
    :param df: Input DataFrame containing the data.
    :param save_path: Path to save the generated diagram (optional).
    """
    # Calculate the correlation matrix
    correlation_matrix = df.corr()
    
    # Extract feature names
    feature_names = df.columns
    
    # Plot the Chord diagram
    chord_diagram(mat=correlation_matrix, names=feature_names, rotate_names=90, alpha=0.7)
    
    # Save the diagram
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    # Display the diagram
    plt.show()

# Main function
if __name__ == "__main__":
    # Load the CSV file (replace with your CSV file path)
    csv_file_path = "your CSV file path/demo.csv"
    df = load_csv(csv_file_path)
    
    # Plot the Chord diagram and save it (replace with your save path)
    save_path = " your save path/demo_chord.png"
    plot_chord_diagram(df, save_path)