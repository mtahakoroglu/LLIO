import os
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import copy

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.stats import skew, kurtosis, iqr
from scipy.signal import welch, hilbert, cwt, ricker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

FS = 200
# Model mimarisi (training kodundakiyle aynı olmalı)
class IMUModel(nn.Module):
    def __init__(self, input_dim):
        super(IMUModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            256,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )
        self.bn = nn.BatchNorm1d(512)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.bn(lstm_out)
        return self.fc(lstm_out)

def extract_wavelet_features(stride_data):
    feats = []
    widths = np.arange(1, 31)
    for ch in range(3):
        sig = stride_data[:, ch]
        cwtmatr = cwt(sig, ricker, widths)
        abs_cwt = np.abs(cwtmatr)
        feats.extend([
            np.mean(abs_cwt), np.max(abs_cwt), np.min(abs_cwt)
        ])
    return feats


def extract_envelope_features(stride_data):
    feats = []
    for ch in range(3):
        sig = stride_data[:, ch]
        amp_env = np.abs(hilbert(sig))
        feats.extend([
            np.mean(amp_env), np.std(amp_env), np.max(amp_env), np.min(amp_env)
        ])

        phase = np.unwrap(np.angle(hilbert(sig)))
        freq_env = np.diff(phase) / (2 * np.pi)
        feats.extend([
            np.mean(freq_env), np.std(freq_env), np.max(freq_env), np.min(freq_env)
        ])
    return feats


def extract_stride_features(stride_data):
    time_feats = extract_time_domain_features(stride_data)
    freq_feats = extract_frequency_domain_features(stride_data)
    wavelet_feats = extract_wavelet_features(stride_data)
    envelope_feats = extract_envelope_features(stride_data)

    acc_norm = np.linalg.norm(stride_data, axis=1)
    peakAcc = np.max(acc_norm) if len(acc_norm) > 0 else 0
    stride_time = stride_data.shape[0] / FS

    return time_feats + freq_feats + wavelet_feats + envelope_feats + [peakAcc, stride_time]


def extract_time_domain_features(stride_data):
    feats = []
    for ch in range(3):
        sig = stride_data[:, ch]
        feats.extend([
            np.mean(sig), np.std(sig), np.max(sig), np.min(sig),
            skew(sig), kurtosis(sig), iqr(sig), np.median(sig),
            np.sum(sig ** 2) / len(sig), np.var(sig)
        ])
    return feats


def extract_frequency_domain_features(stride_data, fs=200):
    feats = []
    for ch in range(3):
        sig = stride_data[:, ch]
        f, Pxx = welch(sig, fs=fs, nperseg=256)
        feats.extend([
            np.sum(Pxx), np.mean(Pxx), np.max(Pxx), np.median(Pxx),
            np.quantile(Pxx, 0.1), np.quantile(Pxx, 0.9)
        ])
    return feats

def rotate_trajectory(trajectory, theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return trajectory @ rotation_matrix.T

# Veri yükleme ve ön işleme
def process_mat_file(file_path, feat_scaler):
    data = loadmat(file_path)
    expID, stride_idx, GCP, acc_n = data["expID"].item(), data["strideIndex"].flatten(), data["GCP_wcf"], data["acc_n"]
    theta = data["theta"].flatten()[0]
    pyshoe_trajectory = data["pyshoeTrajectory"]
    pyshoe_trajectory_wcf = np.squeeze(rotate_trajectory(pyshoe_trajectory, -theta)) # ncf to wcf
    pyshoe_trajectory_wcf[:,1] = -pyshoe_trajectory_wcf[:,1] # change made by mtahakoroglu to match with GT alignment
    # take only stride_stride_idx strides from pyshoe_trajectory_wcf
    pyshoe_trajectory_wcf = pyshoe_trajectory_wcf[stride_idx, :]

    X_list = []
    valid_indices = []

    for i in range(len(stride_idx) - 1):
        s, e = stride_idx[i], stride_idx[i + 1]
        seg = acc_n[s:e, :]
        if seg.shape[0] < 10:
            continue

        feats = extract_stride_features(seg)
        X_list.append(feats)
        valid_indices.append(i)

    # GCP'yi stride sayısına göre filtrele
    # filtered_GCP = GCP[[0] + [i + 1 for i in valid_indices]]
    return np.array(X_list), expID, theta, GCP, pyshoe_trajectory_wcf


def predict_and_plot(data_dir):
    # Cihaz ve model yapılandırması
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Yükleme: scalerlar, parametreler ve model mimarisi
    load_models_directory = "results/LSTM-model/"
    # Scaler'ları yükle
    feat_scaler = joblib.load(f"{load_models_directory}feat_scaler.pkl")
    dist_scaler = joblib.load(f"{load_models_directory}dist_scaler.pkl")
    head_scaler = joblib.load(f"{load_models_directory}heading_scaler.pkl")

    # Modelleri yükle
    # Get input dimension from the scaler's quantiles shape
    if hasattr(feat_scaler, 'n_features_in_'):
        input_dim = feat_scaler.n_features_in_
    else:
        input_dim = feat_scaler.quantiles_.shape[1]
    
    model_dist = IMUModel(input_dim).to(device)
    model_head = IMUModel(input_dim).to(device)

    model_dist.load_state_dict(torch.load(f"{load_models_directory}improved_dist_model.pth", map_location=device))
    model_head.load_state_dict(torch.load(f"{load_models_directory}improved_heading_model.pth", map_location=device))

    model_dist.eval()
    model_head.eval()

    # Tüm MAT dosyalarını işle
    for fname in os.listdir(data_dir):
        if not fname.endswith(".mat"):
            continue

        print(f"\nProcessing {fname}...")
        file_path = os.path.join(data_dir, fname)

        # Veriyi yükle ve özellikleri çıkar
        X, expID, theta, GCP, pyshoe_trajectory_wcf = process_mat_file(file_path, feat_scaler)
        if len(X) == 0:
            print(f"No valid strides in {fname}, skipping...")
            continue

        # Ölçeklendirme
        X_scaled = feat_scaler.transform(X)

        # Tahmin yap
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            dist_pred = model_dist(X_tensor).cpu().numpy()
            head_pred = model_head(X_tensor).cpu().numpy()

        # Ölçeklendirmeyi geri al
        dist_pred = dist_scaler.inverse_transform(dist_pred)
        head_pred = head_scaler.inverse_transform(head_pred)

        # Trajectory hesapla
        x_pred = [0.0]
        y_pred = [0.0]
        for d, h in zip(dist_pred, head_pred):
            dx = d * np.cos(h)
            dy = d * np.sin(h)
            x_pred.append(x_pred[-1] + dx)
            y_pred.append(y_pred[-1] + dy)

        # now change of basis from ncf to wcf
        prediction = np.column_stack((x_pred, y_pred))
        prediction_wcf = np.squeeze(rotate_trajectory(prediction, -theta))
        prediction_wcf[:,1] = -prediction_wcf[:,1] # change made by mtahakoroglu to match with GT alignment

        plt.figure(figsize=(10, 6))
        plt.scatter(GCP[:,0], GCP[:,1], c='orange', marker='s', s=50, edgecolors='k', label='GCP')
        plt.plot(GCP[:,0], GCP[:,1], 'orange', linestyle='--', linewidth=1.5)
        plt.plot(prediction_wcf[:,0], prediction_wcf[:,1], 'b.-', linewidth=2, label='Feature-based LSTM')
        # if expID not in [1, 51, 52, 53, 54]:
        if expID not in [51, 53, 54, 65, 66, 68, 75]:
            plt.plot(pyshoe_trajectory_wcf[:,0], pyshoe_trajectory_wcf[:,1], 'r.-', linewidth=1.5, label='PyShoe (LSTM)')
        # plt.scatter(x_real[0], y_real[0], c='r', marker='s', edgecolors='k', s=50, label='Initial Position')
        # plt.scatter(x_pred, y_pred, c='b', label='LLIO (stride-wise)')
        plt.title(f"{fname.split('_LLIO.mat')[0]}", fontsize=16)
        plt.xlabel("X [m]", fontsize=16)
        plt.ylabel("Y [m]", fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.axis('equal')
        plt.tight_layout()
        plt.subplots_adjust(left=0.097)  # Increase left margin specifically
        plt.xticks(fontsize=16)  # Makes x-axis numbers bigger
        plt.yticks(fontsize=16)  # Makes y-axis numbers bigger
        plt.legend(fontsize=16)
        plt.savefig(f"results/figs/feature-based-LSTM/feature_based_lstm_exp_{expID}.png", dpi=300)
        # With these lines:
        plt.pause(1)  # Show the figure for 1 second
        plt.close()   # Close the figure automatically


if __name__ == "__main__":
    data_directory = "data/testing"
    predict_and_plot(data_directory)