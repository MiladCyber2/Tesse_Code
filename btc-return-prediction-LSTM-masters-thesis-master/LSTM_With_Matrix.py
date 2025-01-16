import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

from plotting import *
from functions import *
from import_data import data_scaled, data_scalers, data_raw
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



WINDOW = 50
HORIZON = 1
SPLIT = 0.8
EPOCHS = 100
TRAIN_MODEL = True
ACTIVE_MODEL = "hp2"
TMP_FLDR = (
    "C:/Backup master data/models/tmp/"  # Temporary model files "saved_models/tmp/"
)



def save_metrics_to_file(y_true, y_pred, file_path):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    }
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(file_path, index=False)
    print(f"Metrics saved to {file_path}")

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()


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

    print(y_test[:, 0, 0].shape)
    print(y_test[1:4, 0, 0])

    # define the model
    global n_features
    n_features = len(data.keys())
    from models import model_dict

    model = model_dict[ACTIVE_MODEL]["model"]

    #دستکاری با کد زیریmodel.build(input_shape=x_train.shape)
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    # stop early if model is not improving for patience=n epochs
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=25, restore_best_weights=True
    )
    ten_callback = tf.keras.callbacks.ModelCheckpoint(
       #دستکاری با کد زیری filepath=TMP_FLDR + "model.{epoch:04d}.h5", 
      #برای وزن ها بود و دستکاری کده بودم filepath=TMP_FLDR + "model.{epoch:04d}.weights.h5", 
       # save_best_only=False,
       # save_weights_only=True,
        filepath=TMP_FLDR + "model.{epoch:04d}.keras",
    save_best_only=False,
    save_weights_only=False,
    )

    if TRAIN_MODEL:
        model = train_model(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            EPOCHS,
            [es_callback, ten_callback],
            batch_size=model_dict[ACTIVE_MODEL]["batch_size"],
        )
    else:
        model = tf.keras.models.load_model("C:/Users/Syber/Desktop/VS_code/code/btc-return-prediction-LSTM-masters-thesis-master/saved_models/modelbuilder_model.h5")
        print("Model loaded from saved model (C:/Users/Syber/Desktop/VS_code/code/btc-return-prediction-LSTM-masters-thesis-master/saved_models/modelbuilder_model.h5).")

    epoch_count = len(model.history.epoch)
    model_history = pd.DataFrame(model.history.history)
    model_history["epoch"] = model.history.epoch
    num_epochs = model_history.shape[0]
    # plot_model_loss_training(model)

    # get the models predicted values
    # pred_train = data_scalers["btc_logreturn"].inverse_transform(model.predict(x_train))
    pred_test = data_scalers["btc_logreturn"].inverse_transform(model.predict(x_test))
    y_test_unscaled = data_scalers["btc_logreturn"].inverse_transform(y_test[:, :, 0])
    

    # Binarize the predictions and true values for classification metrics
    y_test_binary = (y_test_unscaled > 0).astype(int)
    y_pred_binary = (pred_test > 0).astype(int)

    # Save metrics to file
    save_metrics_to_file(y_test_binary, y_pred_binary, 'C:/Users/Syber/Desktop/VS_code/code/Metrics/LSTM_metrics.csv')

    # Save confusion matrix data
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
    cm_df.to_csv('C:/Users/Syber/Desktop/VS_code/code/Metrics/LSTM_confusion_matrix_data.csv')
    print("Confusion matrix saved to C:/Users/Syber/Desktop/VS_code/code/Metrics/LSTM_confusion_matrix_data.csv")

    # Plot confusion matrix
    plot_confusion_matrix(cm)

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
        TMP_FLDR,
        "simple",
    )
    clear_tmp_saved_models(TMP_FLDR)
    return


if __name__ == "__main__":
    main()
