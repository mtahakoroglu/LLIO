import os
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, iqr
import torch
from torch import nn
from scipy.io import loadmat
from scipy.signal import welch, hilbert, cwt, ricker
import random

warnings.filterwarnings("ignore", category=UserWarning)


class IMUModel(nn.Module):
    def __init__(self, input_dim, hidden_size=256, num_layers=2, dropout=0.3):
        super(IMUModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0
        )
        self.bn = nn.BatchNorm1d(2 * hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.bn(lstm_out)
        return self.fc(lstm_out)


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


def extract_wavelet_features(stride_data):
    feats = []
    widths = np.arange(1, 31)
    for ch in range(3):
        sig = stride_data[:, ch]
        cwtmatr = cwt(sig, ricker, widths)
        abs_cwt = np.abs(cwtmatr)
        feats.extend([np.mean(abs_cwt), np.max(abs_cwt), np.min(abs_cwt)])
    return feats


def extract_envelope_features(stride_data):
    feats = []
    for ch in range(3):
        sig = stride_data[:, ch]
        amp_env = np.abs(hilbert(sig))
        feats.extend([np.mean(amp_env), np.std(amp_env), np.max(amp_env), np.min(amp_env)])
        phase = np.unwrap(np.angle(hilbert(sig)))
        freq_env = np.diff(phase) / (2 * np.pi)
        feats.extend([np.mean(freq_env), np.std(freq_env), np.max(freq_env), np.min(freq_env)])
    return feats


def extract_stride_features(stride_data):
    time_feats = extract_time_domain_features(stride_data)
    freq_feats = extract_frequency_domain_features(stride_data)
    wavelet_feats = extract_wavelet_features(stride_data)
    envelope_feats = extract_envelope_features(stride_data)
    acc_norm = np.linalg.norm(stride_data, axis=1)
    peakAcc = np.max(acc_norm) if len(acc_norm) > 0 else 0
    stride_time = stride_data.shape[0] / 200
    return time_feats + freq_feats + wavelet_feats + envelope_feats + [peakAcc, stride_time]

def rotate_trajectory(trajectory, theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return trajectory @ rotation_matrix.T

def process_mat_file(file_path, feat_scaler):
    data = loadmat(file_path)
    stride_idx = data["strideIndex"].flatten()
    GCP = data["GCP"]
    acc_n = data["acc_n"]
    
    theta = data["theta"].flatten()[0]
    GCP_wcf = data["GCP_wcf"] # GCP_wcf is already aligned in mat file so no need for rotation about x axis
    pyshoe_trajectory = data["pyshoeTrajectory"]
    pyshoe_trajectory_wcf = np.squeeze(rotate_trajectory(pyshoe_trajectory, -theta)) # ncf to wcf
    pyshoe_trajectory_wcf[:,1] = -pyshoe_trajectory_wcf[:,1] # change made by mtahakoroglu to match with GT alignment
    
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
    # GCP dizisini, örneklerin sırasına göre filtreliyoruz.
    filtered_GCP = GCP[[0] + [i + 1 for i in valid_indices]]
    return np.array(X_list), filtered_GCP, theta, GCP_wcf, pyshoe_trajectory_wcf


def predict_and_plot(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Yükleme: scalerlar, parametreler ve model mimarisi
    load_models_directory = "results/LSTM-model/"
    feat_scaler = joblib.load(f"{load_models_directory}best_feat_scaler.pkl")
    dist_scaler = joblib.load(f"{load_models_directory}best_dist_scaler.pkl")
    head_scaler = joblib.load(f"{load_models_directory}best_heading_scaler.pkl")
    best_params = joblib.load(f"{load_models_directory}best_params.pkl")
    
    input_dim = feat_scaler.quantiles_.shape[1]
    model_dist = IMUModel(
        input_dim=input_dim,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(device)

    model_head = IMUModel(
        input_dim=input_dim,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(device)

    model_dist.load_state_dict(torch.load(f"{load_models_directory}best_dist_model.pth", map_location=device))
    model_head.load_state_dict(torch.load(f"{load_models_directory}best_heading_model.pth", map_location=device))

    model_dist.eval()
    model_head.eval()

    for fname in os.listdir(data_dir):
        if not fname.endswith(".mat"):
            continue

        print(f"\nProcessing {fname}...")
        file_path = os.path.join(data_dir, fname)
        X, GCP, theta, GCP_wcf, pyshoe_trajectory_wcf = process_mat_file(file_path, feat_scaler)

        if len(X) == 0:
            print(f"No valid strides in {fname}, skipping...")
            continue

        X_scaled = feat_scaler.transform(X)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            dist_pred = model_dist(X_tensor).cpu().numpy()
            head_pred = model_head(X_tensor).cpu().numpy()

        dist_pred = dist_scaler.inverse_transform(dist_pred)
        head_pred = head_scaler.inverse_transform(head_pred)

        x_pred = [0.0]
        y_pred = [0.0]
        for d, h in zip(dist_pred, head_pred):
            dx = d * np.cos(h)
            dy = d * np.sin(h)
            x_pred.append(x_pred[-1] + dx)
            y_pred.append(y_pred[-1] + dy)
            
        LLIO = np.column_stack((x_pred, y_pred))
        LLIO_wcf = np.squeeze(rotate_trajectory(LLIO, -theta))
        LLIO_wcf[:,1] = -LLIO_wcf[:,1] # change made by mtahakoroglu to match with GT alignment

        # LLIO prediction in NCS
        x_real = GCP[:, 0] - GCP[0, 0]
        y_real = GCP[:, 1] - GCP[0, 1]

        plt.figure(figsize=(10, 6))
        # plt.scatter(x_real, y_real, c='r', marker='s', s=50, edgecolors='k', label='GCP')
        plt.scatter(GCP_wcf[:,0], GCP_wcf[:,1], c='r', marker='s', s=50, edgecolors='k', label='GCP')
        # plt.plot(x_real, y_real, 'r--', linewidth=1.5)
        plt.plot(GCP_wcf[:,0], GCP_wcf[:,1], 'r--', linewidth=1.5)
        # plt.plot(x_pred, y_pred, 'b.-', linewidth=1.5, label='LLIO')
        plt.plot(LLIO_wcf[:,0], LLIO_wcf[:,1], 'b.-', linewidth=2, label='LLIO (LSTM)')
        plt.plot(pyshoe_trajectory_wcf[:,0], pyshoe_trajectory_wcf[:,1], 'g-', linewidth=1.5, label='PyShoe (LSTM)')
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
        plt.savefig(f"results/figs/seed_10.png", dpi=500)
        plt.show()


if __name__ == "__main__":
    data_directory = "data/testing"
    predict_and_plot(data_directory)