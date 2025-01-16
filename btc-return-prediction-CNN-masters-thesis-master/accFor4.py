import pandas as pd
import matplotlib.pyplot as plt

# Load metrics from CSV files
metrics_main = pd.read_csv('C:/Users/Syber/Desktop/VS_code/code/Metrics/metrics_main.csv')
metrics_cnn_lstm = pd.read_csv('C:/Users/Syber/Desktop/VS_code/code/Metrics/metrics_CNN_LSTM.csv')
metrics_mh_cnn_lstm = pd.read_csv('C:/Users/Syber/Desktop/VS_code/code/Metrics/metrics_MH_CNN_LSTM.csv')
metrics_lstm = pd.read_csv('C:/Users/Syber/Desktop/VS_code/code/Metrics/metrics_LSTM.csv')

# Combine metrics into a single DataFrame
metrics_combined = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'CNN': [metrics_main['accuracy'][0], metrics_main['precision'][0], metrics_main['recall'][0], metrics_main['f1'][0]],
    'CNN_LSTM': [metrics_cnn_lstm['accuracy'][0], metrics_cnn_lstm['precision'][0], metrics_cnn_lstm['recall'][0], metrics_cnn_lstm['f1'][0]],
    'MH_CNN_LSTM': [metrics_mh_cnn_lstm['accuracy'][0], metrics_mh_cnn_lstm['precision'][0], metrics_mh_cnn_lstm['recall'][0], metrics_mh_cnn_lstm['f1'][0]],
    'LSTM': [metrics_lstm['accuracy'][0], metrics_lstm['precision'][0], metrics_lstm['recall'][0], metrics_lstm['f1'][0]]
})

# Plot the metrics
metrics_combined.set_index('Metric').plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Value')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.show()
