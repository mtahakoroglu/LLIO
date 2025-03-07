import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.io import loadmat
from scipy.stats import skew, kurtosis, iqr
from scipy.signal import welch

##############################################################################
# 1) FEATURE FUNCTIONS (Aynı)
##############################################################################
def extract_time_domain_features(stride_data):
    feats = []
    for ch in range(stride_data.shape[1]):
        sig = stride_data[:, ch]
        feats.extend([
            np.mean(sig),
            np.std(sig),
            np.max(sig),
            np.min(sig),
            skew(sig),
            kurtosis(sig),
            iqr(sig),
            np.median(sig),
            np.sum(sig**2)/len(sig),
            np.var(sig)
        ])
    return feats

def extract_frequency_domain_features(stride_data, fs=200):
    feats = []
    n_ch = stride_data.shape[1]
    for ch in range(n_ch):
        sig = stride_data[:, ch]
        f, Pxx = welch(sig, fs=fs, nperseg=256)
        feats.append(np.sum(Pxx))
        feats.append(np.mean(Pxx))
        feats.append(np.max(Pxx))
        feats.append(np.median(Pxx))
    return feats

def extract_autocorr_features(stride_data, lags=5):
    feats = []
    n_ch = stride_data.shape[1]
    for ch in range(n_ch):
        sig = stride_data[:, ch]
        sig -= np.mean(sig)
        ac = np.correlate(sig, sig, mode='full')
        ac = ac[len(ac)//2:]
        if ac[0] != 0:
            ac = ac/ac[0]
        for lag in range(1,lags+1):
            if lag < len(ac):
                feats.append(ac[lag])
            else:
                feats.append(0)
    return feats

def extract_stride_features(stride_data, fs=200):
    time_feats = extract_time_domain_features(stride_data)
    freq_feats = extract_frequency_domain_features(stride_data, fs=fs)
    ac_feats   = extract_autocorr_features(stride_data, lags=5)

    acc_norm = np.linalg.norm(stride_data, axis=1)
    peakAcc = np.max(acc_norm)
    dt = 1.0/fs
    stride_time = stride_data.shape[0]*dt

    feats = time_feats + freq_feats + ac_feats + [peakAcc, stride_time]
    return feats

##############################################################################
# 2) .mat Parse => (X_feats, dist_true, head_true)
##############################################################################
def parse_mat_file(file_path):
    mat = loadmat(file_path)
    stride_idx = mat["strideIndex"].flatten()
    GCP = mat["GCP"]
    acc_n = mat["acc_n"]

    n_strides = min(len(stride_idx)-1, GCP.shape[0])
    X_list = []
    dist_list = []
    head_list = []

    for i in range(n_strides):
        s = stride_idx[i]
        e = stride_idx[i+1]
        seg = acc_n[s:e, :]
        if seg.shape[0]<2:
            continue
        feats = extract_stride_features(seg, fs=200)

        if i < n_strides-1:
            dx = GCP[i+1,0] - GCP[i,0]
            dy = GCP[i+1,1] - GCP[i,1]
        else:
            dx, dy = 0, 0
        dist_ = np.sqrt(dx**2 + dy**2)
        head_ = np.arctan2(dy, dx)

        X_list.append(feats)
        dist_list.append(dist_)
        head_list.append(head_)

    return np.array(X_list), np.array(dist_list), np.array(head_list)

##############################################################################
# 3) Tahmin + Plot
##############################################################################

GB_MODEL_PATH = "results/GB-model/"
DIST_MODEL_PATH = GB_MODEL_PATH + "dist_model.pkl"
HEADING_MODEL_PATH = GB_MODEL_PATH + "heading_model.pkl"
DIST_SCALER_PATH = GB_MODEL_PATH + "dist_scaler.pkl"
HEADING_SCALER_PATH = GB_MODEL_PATH + "heading_scaler.pkl"
FEATCOLS_PATH = GB_MODEL_PATH + "feat_cols.pkl"

def predict_and_plot(data_dir, save_plots=False):
    # Modelleri yükle
    dist_model = joblib.load(DIST_MODEL_PATH)
    head_model = joblib.load(HEADING_MODEL_PATH)
    dist_scaler = joblib.load(DIST_SCALER_PATH)
    head_scaler = joblib.load(HEADING_SCALER_PATH)
    feat_cols   = joblib.load(FEATCOLS_PATH)

    for fname in os.listdir(data_dir):
        if fname.endswith(".mat"):
            path_ = os.path.join(data_dir, fname)
            X_arr, dist_true, head_true = parse_mat_file(path_)
            if X_arr.shape[0] == 0:
                continue

            df_feat = pd.DataFrame(X_arr, columns=feat_cols)

            # DISTANCE Tahmini
            X_d_sc = dist_scaler.transform(df_feat.values)
            dist_pred = dist_model.predict(X_d_sc)

            # HEADING Tahmini
            X_h_sc = head_scaler.transform(df_feat.values)
            head_pred = head_model.predict(X_h_sc)

            # 1) Scatter Plot
            plt.figure(figsize=(10,4))
            # dist scatter
            plt.subplot(1,2,1)
            plt.scatter(dist_true, dist_pred, alpha=0.5)
            mx_d = max(dist_true.max(), dist_pred.max())
            plt.plot([0,mx_d],[0,mx_d],'r--')
            plt.xlabel("True Dist")
            plt.ylabel("Pred Dist")
            plt.title(f"{fname} - Distance")

            # heading scatter
            plt.subplot(1,2,2)
            mn_h = min(head_true.min(), head_pred.min())
            mx_h = max(head_true.max(), head_pred.max())
            plt.scatter(head_true, head_pred, alpha=0.5, color='orange')
            plt.plot([mn_h,mx_h],[mn_h,mx_h],'r--')
            plt.xlabel("True Heading")
            plt.ylabel("Pred Heading")
            plt.title(f"{fname} - Heading")
            plt.tight_layout()

            if save_plots:
                scatter_path = os.path.join(data_dir, f"scatter_{fname}.png")
                plt.savefig(scatter_path, dpi=150)
                plt.close()
            else:
                plt.show()

            # 2) Adım adım konum rekonstrüksiyon
            # Pred
            x_pred, y_pred = [0],[0]
            for i in range(len(dist_pred)):
                d = dist_pred[i]
                h = head_pred[i]
                x_new = x_pred[-1] + d*np.cos(h)
                y_new = y_pred[-1] + d*np.sin(h)
                x_pred.append(x_new)
                y_pred.append(y_new)

            # True
            x_true, y_true = [0],[0]
            for i in range(len(dist_true)):
                d = dist_true[i]
                h = head_true[i]
                x_new = x_true[-1] + d*np.cos(h)
                y_new = y_true[-1] + d*np.sin(h)
                x_true.append(x_new)
                y_true.append(y_new)

            plt.figure()
            plt.plot(x_true, y_true, 'o-', label='True Traj')
            plt.plot(x_pred, y_pred, 'x--', label='Pred Traj')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.title(f"Trajectory: {fname}")
            plt.grid(True)
            if save_plots:
                traj_path = os.path.join(data_dir, f"traj_{fname}.png")
                plt.savefig(traj_path, dpi=150)
                plt.close()
            else:
                plt.show()

if __name__ == "__main__":
    data_directory = r"data\LLIO_training_data"
    predict_and_plot(data_directory, save_plots=False)
