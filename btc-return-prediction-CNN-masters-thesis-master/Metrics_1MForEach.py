import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load metrics from CSV files
metrics_main = pd.read_csv('C:/Users/Syber/Desktop/VS_code/code/Metrics/CNN_metrics.csv')
metrics_cnn_lstm = pd.read_csv('C:/Users/Syber/Desktop/VS_code/code/Metrics/CNN_LSTM_metrics.csv')
metrics_mh_cnn_lstm = pd.read_csv('C:/Users/Syber/Desktop/VS_code/code/Metrics/MH-CNN-LSTM_metrics.csv')
metrics_lstm = pd.read_csv('C:/Users/Syber/Desktop/VS_code/code/Metrics/LSTM_metrics.csv')

# Combine metrics into a single DataFrame
metrics_combined = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'CNN': [metrics_main['Accuracy'][0], metrics_main['Precision'][0], metrics_main['Recall'][0], metrics_main['F1 Score'][0]],
    'CNN_LSTM': [metrics_cnn_lstm['Accuracy'][0], metrics_cnn_lstm['Precision'][0], metrics_cnn_lstm['Recall'][0], metrics_cnn_lstm['F1 Score'][0]],
    'MH_CNN_LSTM': [metrics_mh_cnn_lstm['Accuracy'][0], metrics_mh_cnn_lstm['Precision'][0], metrics_mh_cnn_lstm['Recall'][0], metrics_mh_cnn_lstm['F1 Score'][0]],
    'LSTM': [metrics_lstm['Accuracy'][0], metrics_lstm['Precision'][0], metrics_lstm['Recall'][0], metrics_lstm['F1 Score'][0]]
})

# Plot each metric in a separate figure
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Define a list of colors for each model

for i, metric in enumerate(metrics_combined['Metric']):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define the bar width and positions
    bar_width = 0.2
    bar_positions = np.arange(len(metrics_combined.columns) - 1) * bar_width
    
    # Plot each model's metric with the specified color and position
    for j, model in enumerate(metrics_combined.columns[1:]):
        ax.bar(bar_positions[j], metrics_combined[model][i], width=bar_width, color=colors[j], edgecolor='black', label=model)
    
    # Set the x-axis labels and positions
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(metrics_combined.columns[1:])
    
    # Set the title and labels
    plt.title(f'{metric} Comparison')
    plt.ylabel(metric)
    plt.ylim(0, 1)
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Adjust the layout to ensure the first bar has some margin
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    
    plt.show()
