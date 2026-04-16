import numpy as np
import joblib
import tensorflow as tf


# ============================
# 1. USER COUNT (RandomForest)
# ============================
def extract_csi_features(npy_path):
    csi_array = np.load(npy_path)

    features = []
    for tx in range(csi_array.shape[1]):
        for rx in range(csi_array.shape[2]):
            for sc in range(csi_array.shape[3]):
                signal = csi_array[:, tx, rx, sc]
                features.extend([
                    signal.mean(),
                    signal.std(),
                    signal.min(),
                    signal.max()
                ])

    return np.array(features).reshape(1, -1)


def predict_user_count(npy_path, rf_model):
    features = extract_csi_features(npy_path)
    pred = rf_model.predict(features)
    return int(pred[0])


# ============================
# 2. ACTIVITY (CNN + LSTM)
# ============================
def create_windows(npy_path, window_size=256, step=64):
    csi_array = np.load(npy_path)

    # flatten like training
    csi_flat = csi_array.reshape(csi_array.shape[0], -1)

    windows = []
    for i in range(0, len(csi_flat) - window_size, step):
        windows.append(csi_flat[i:i + window_size])

    return np.array(windows)


def preprocess_windows(windows, scaler):
    shape = windows.shape

    windows_reshaped = windows.reshape(-1, windows.shape[-1])
    windows_scaled = scaler.transform(windows_reshaped)

    return windows_scaled.reshape(shape)


def predict_activity(npy_path, model, scaler, label_encoder):
    windows = create_windows(npy_path)

    if len(windows) == 0:
        return "Not enough data"

    windows = preprocess_windows(windows, scaler)

    preds = model.predict(windows, verbose=0)
    classes = np.argmax(preds, axis=1)

    # majority voting
    final_class = np.bincount(classes).argmax()
    activity = label_encoder.inverse_transform([final_class])[0]

    return activity


# ============================
# 3. UNIFIED PIPELINE
# ============================
def run_inference(
    npy_path,
    rf_model_path,
    activity_model_path,
    scaler_path,
    label_encoder_path
):
    # Load everything
    rf_model = joblib.load(rf_model_path)
    activity_model = tf.keras.models.load_model(activity_model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)

    # Predictions
    user_count = predict_user_count(npy_path, rf_model)
    activity = predict_activity(npy_path, activity_model, scaler, label_encoder)

    # Final output
    print("\n===== FINAL RESULT =====")
    print(f"👥 Number of People   : {user_count}")
    print(f"🏃 Activity Detected : {activity}")
    print("=======================\n")

    return {
        "num_people": user_count,
        "activity": activity
    }


# ============================
# 4. Example usage
# ============================
if __name__ == "__main__":
    result = run_inference(
        npy_path="act_103_11.npy",
        rf_model_path="no_of_people_model.pkl",
        activity_model_path="activity_model.h5",
        scaler_path="activity_scaler.pkl",
        label_encoder_path="activity_label_encoder.pkl"
    )