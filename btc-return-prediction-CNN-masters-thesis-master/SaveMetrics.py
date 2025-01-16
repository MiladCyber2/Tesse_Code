import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from functions import *
from import_data import data_scaled, data_scalers, data_raw

WINDOW = 50
HORIZON = 1
SPLIT = 0.8
EPOCHS = 5
BATCH_SIZE = 32

def plot_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(10, 5))
    plt.bar(names, values, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1)
    for i in range(len(values)):
        plt.text(i, values[i] + 0.01, f'{values[i]:.2f}', ha='center')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    return accuracy, precision, recall, f1

def save_metrics_to_csv(metrics, filename):
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(filename, index=False)

def main():
    ################### Data pre processing ######################
    data = data_scaled
    x_train, y_train, x_test, y_test = create_train_test(
        data,
        {"btc_logreturn": data["btc_logreturn"]},
        window=WINDOW,
        horizon=HORIZON,
        split=SPLIT,
    )

    n_features = len(data.keys())
    print(x_test.shape)

    print(y_test[:, 0, 0].shape)
    print(y_test[1:4, 0, 0])

    # define the CNN model
    # define the input tensor
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=[WINDOW, n_features]),
            # define the convolutional layers
            tf.keras.layers.Conv1D(
                filters=32, kernel_size=3, padding="same", activation=tf.nn.relu
            ),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, padding="same", activation=tf.nn.relu
            ),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            tf.keras.layers.Conv1D(
                filters=128, kernel_size=3, padding="same", activation=tf.nn.relu
            ),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
            # flatten the output of the convolutional layers
            tf.keras.layers.Flatten(),
            # define the fully-connected layers
            tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
            # define the output layer
            tf.keras.layers.Dense(units=1),
        ],
    )
    # model.build(input_shape=x_train.shape)
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file="model_shape_CNN.png", show_shapes=True
    )
    
    #
    # print(model.summary())
    # stop early if model is not improving for patience=n epochs
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=15, restore_best_weights=True
    )
    model.compile(loss="mae", optimizer="adam", metrics=["mse", "mae"])
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[es_callback],
        shuffle=False,
    )

    epoch_count = len(model.history.epoch)
    model_history = pd.DataFrame(model.history.history)
    model_history["epoch"] = model.history.epoch
    num_epochs = model_history.shape[0]

    # get the models predicted values
    pred_test = data_scalers["btc_logreturn"].inverse_transform(model.predict(x_test))
    y_test_unscaled = data_scalers["btc_logreturn"].inverse_transform(y_test[:, :, 0])

    # Binarize the predictions and true values for classification metrics
    y_pred_binary = (pred_test > 0.5).astype(int)
    y_test_binary = (y_test_unscaled > 0.5).astype(int)

    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_metrics(y_test_binary, y_pred_binary)

    # Save metrics to a file
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    save_metrics_to_csv(metrics, 'C:/Users/Syber/Desktop/VS_code/code/Metrics/metrics_main.csv')
    # Plot the metrics
    plot_metrics(y_test_binary, y_pred_binary)

    # Plot the confusion matrix
    plot_confusion_matrix(y_test_binary, y_pred_binary)

    model_stats(model.name, pred_test, x_test, y_test_unscaled, model, epoch_count)
    print_model_statistics(
        model_statistics(pred_test, x_test, y_test, model), x_test, y_test, model
    )

    save_loss_history(
        num_epochs, model_history["loss"], model_history["val_loss"], "last_model"
    )
    main_plot(
        WINDOW,
        HORIZON,
        EPOCHS,
        model,
        x_train,
        num_epochs,
        model_history,
        data,
        data_scalers,
        pred_test,
    )
    pred_plot(
        "CNN",
        WINDOW,
        HORIZON,
        EPOCHS,
        model,
        x_train,
        num_epochs,
        model_history,
        data,
        data_scalers,
        pred_test,
    )
    return

if __name__ == "__main__":
    main()
