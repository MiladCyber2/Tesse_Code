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

# Plot the metrics with different colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Define a list of colors for each metric

fig, ax = plt.subplots(figsize=(12, 6))

# Define the bar width and positions
bar_width = 0.2
bar_positions = np.arange(len(metrics_combined.columns) - 1) * (len(metrics_combined) + 1) * bar_width

# Plot each model's metrics with the specified color and position
for i, model in enumerate(metrics_combined.columns[1:]):
    for j, color in enumerate(colors):
        ax.bar(bar_positions[i] + j * bar_width, metrics_combined[model][j], width=bar_width, color=color, edgecolor='black')

# Set the x-axis labels and positions
ax.set_xticks(bar_positions + bar_width * (len(metrics_combined) - 1) / 2)
ax.set_xticklabels(metrics_combined.columns[1:])

# Set the title and labels
plt.title('Model Performance Comparison')
plt.ylabel('Value')
plt.ylim(0, 1)

# Add color legend manually
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
labels = metrics_combined['Metric']
plt.legend(handles, labels, loc='lower right')

# Adjust the layout to ensure the first bar has some margin
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

plt.show()
