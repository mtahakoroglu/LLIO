import os
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import copy
import itertools
from time import time
import random

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy.stats import skew, kurtosis, iqr
from scipy.signal import welch, hilbert, cwt, ricker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

# Reproducibility için seed ayarları
SEED = 10 # 10 is also good
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# GridSearch parametreleri
PARAM_GRID = {
    'learning_rate': [0.0005],
    'batch_size': [32],
    'hidden_size': [256],
    'num_layers': [5],
    'dropout': [0.2]
}

# Sabit parametreler
FS = 200
ADD_SYNTHETIC = True
SYNTH_SAMPLES = 1800
USE_AUGMENT = True
AUG_TIMES = 4
EARLY_STOP_PATIENCE = 15
MAX_EPOCHS = 20000

# =============================================================================
# Dataset ve Model Tanımlamaları
# =============================================================================
class IMUDataset(Dataset):
    def __init__(self, X, y_dist, y_head, feat_scaler=None, dist_scaler=None, head_scaler=None):
        if feat_scaler:
            self.X = feat_scaler.transform(X.astype(np.float32))
        else:
            self.X = X.astype(np.float32)

        if dist_scaler:
            self.y_dist = dist_scaler.transform(y_dist.reshape(-1, 1)).flatten()
        else:
            self.y_dist = y_dist.astype(np.float32)

        if head_scaler:
            self.y_head = head_scaler.transform(y_head.reshape(-1, 1)).flatten()
        else:
            self.y_head = y_head.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.X[idx]),
            'dist': torch.FloatTensor([self.y_dist[idx]]),
            'heading': torch.FloatTensor([self.y_head[idx]])
        }

class IMUModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
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

# =============================================================================
# Özellik Çıkarım Fonksiyonları
# =============================================================================
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
    stride_time = stride_data.shape[0] / FS
    return time_feats + freq_feats + wavelet_feats + envelope_feats + [peakAcc, stride_time]

def parse_mat_file(file_path):
    data = loadmat(file_path)
    stride_idx = data["strideIndex"].flatten()
    GCP = data["GCP"]
    acc_n = data["acc_n"]
    X_list, dist_list, head_list = [], [], []
    for i in range(len(stride_idx) - 1):
        s, e = stride_idx[i], stride_idx[i + 1]
        seg = acc_n[s:e, :]
        if seg.shape[0] < 10:
            continue
        feats = extract_stride_features(seg)
        dx = GCP[i + 1, 0] - GCP[i, 0]
        dy = GCP[i + 1, 1] - GCP[i, 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        heading = np.arctan2(dy, dx)
        X_list.append(feats)
        dist_list.append(dist)
        head_list.append(heading)
    return np.array(X_list), np.array(dist_list), np.array(head_list)

# =============================================================================
# Veri Yükleme: Own ve Vicon Datada Outlier Ayıklama
# =============================================================================
def load_data_from_folder(data_dir, filter_outliers=True, threshold_percentile=98):
    own_files = []
    vicon_files = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".mat"):
            if "SensorConnectData" in fname:
                own_files.append(os.path.join(data_dir, fname))
            else:
                vicon_files.append(os.path.join(data_dir, fname))

    X_list, dist_list, head_list = [], [], []
    own_info_list = []  # own dosyalarındaki (X, D, H) tuple’ları saklamak için

    # Own verilerini yükle
    for file in own_files:
        X, D, H = parse_mat_file(file)
        if len(X) > 0:
            own_info_list.append((file, X, D, H))

    outlier_scaler = None
    outlier_params = None
    if filter_outliers and len(own_info_list) > 0:
        # Tüm own sample’ları birleştirip referans dağılım hesaplanıyor
        all_own_samples = np.vstack([info[1] for info in own_info_list])
        scaler_z = StandardScaler()
        own_z = scaler_z.fit_transform(all_own_samples)
        ref_mean = np.mean(own_z, axis=0)
        cov = np.cov(own_z, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6  # singularlikten kaçınma
        cov_inv = np.linalg.inv(cov)
        own_md = np.array([np.sqrt(np.dot((row - ref_mean).T, np.dot(cov_inv, (row - ref_mean))))
                           for row in own_z])
        threshold = np.percentile(own_md, threshold_percentile)
        print(f"Own veriler için outlier threshold ({threshold_percentile}th percentile): {threshold:.3f}")

        # Own verilerde filtreleme
        for file, X, D, H in own_info_list:
            X_z = scaler_z.transform(X)
            md = np.array([np.sqrt(np.dot((row - ref_mean).T, np.dot(cov_inv, (row - ref_mean))))
                           for row in X_z])
            valid_indices = np.where(md <= threshold)[0]
            if len(valid_indices) > 0:
                print(f"Own file {os.path.basename(file)}: {len(valid_indices)}/{len(X)} sample selected (threshold = {threshold:.3f})")
                X_list.append(X[valid_indices])
                dist_list.append(D[valid_indices])
                head_list.append(H[valid_indices])
            else:
                print(f"Own file {os.path.basename(file)}: No valid samples (threshold = {threshold:.3f})")
        outlier_scaler = scaler_z
        outlier_params = {'ref_mean': ref_mean, 'cov_inv': cov_inv, 'threshold': threshold}
    else:
        for file, X, D, H in own_info_list:
            X_list.append(X)
            dist_list.append(D)
            head_list.append(H)

    # Vicon verilerinin filtrelenmesi (own verilerden elde edilen referans kullanılarak)
    for file in vicon_files:
        X, D, H = parse_mat_file(file)
        if len(X) > 0:
            if outlier_scaler is not None and outlier_params is not None:
                X_z = outlier_scaler.transform(X)
                ref_mean = outlier_params['ref_mean']
                cov_inv = outlier_params['cov_inv']
                threshold = outlier_params['threshold']
                md = np.array([np.sqrt(np.dot((row - ref_mean).T, np.dot(cov_inv, (row - ref_mean))))
                               for row in X_z])
                valid_indices = np.where(md <= threshold)[0]
                if len(valid_indices) > 0:
                    print(f"VICON file {os.path.basename(file)}: {len(valid_indices)}/{len(X)} samples selected (threshold = {threshold:.3f})")
                    X_list.append(X[valid_indices])
                    dist_list.append(D[valid_indices])
                    head_list.append(H[valid_indices])
                else:
                    print(f"VICON file {os.path.basename(file)}: No valid samples (threshold = {threshold:.3f})")
            else:
                X_list.append(X)
                dist_list.append(D)
                head_list.append(H)

    X_all = np.vstack(X_list)
    D_all = np.hstack(dist_list)
    H_all = np.hstack(head_list)
    X_all = np.nan_to_num(X_all)
    X_all[X_all == np.inf] = 0
    X_all[X_all == -np.inf] = 0
    return X_all, D_all, H_all, outlier_scaler, outlier_params

# =============================================================================
# Veri Artırma Fonksiyonları
# =============================================================================
def create_synthetic_data(df, feat_cols, n_samples=300):
    rows = []
    for _ in range(n_samples):
        row = df.sample(n=1).iloc[0].copy()
        for col in feat_cols:
            noise_scale = 0.02 * abs(row[col])
            row[col] += np.random.normal(0, noise_scale)
        row["dist"] *= np.random.uniform(0.95, 1.05)
        row["heading"] += np.random.uniform(-0.02, 0.02)
        row["heading"] = (row["heading"] + np.pi) % (2 * np.pi) - np.pi
        rows.append(row)
    return pd.DataFrame(rows)

def augment_data(df, feat_cols, times=1):
    aug_rows = []
    for _ in range(times):
        for idx in range(len(df)):
            row = df.iloc[idx].copy()
            for col in feat_cols:
                row[col] *= np.random.uniform(0.98, 1.02)
                row[col] += np.random.normal(0, 0.01 * abs(row[col]))
            row["dist"] *= np.random.uniform(0.99, 1.01)
            row["heading"] += np.random.uniform(-0.005, 0.005)
            row["heading"] = (row["heading"] + np.pi) % (2 * np.pi) - np.pi
            aug_rows.append(row)
    return pd.DataFrame(aug_rows)

# =============================================================================
# Model Eğitimi ve Değerlendirme Fonksiyonları
# =============================================================================
def train_model(params, X_train, y_dist_train, y_head_train, X_val, y_dist_val, y_head_val, feat_scaler, dist_scaler, head_scaler, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = IMUDataset(X_train, y_dist_train, y_head_train)
    val_dataset = IMUDataset(X_val, y_dist_val, y_head_val)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'] * 2, shuffle=False)

    model_dist = IMUModel(
        input_dim=input_dim,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)

    model_head = IMUModel(
        input_dim=input_dim,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)

    criterion = nn.HuberLoss(delta=0.5)
    opt_dist = optim.AdamW(model_dist.parameters(), lr=params['learning_rate'], weight_decay=1e-4)
    opt_head = optim.AdamW(model_head.parameters(), lr=params['learning_rate'], weight_decay=1e-4)

    best_val_mse = float('inf')
    best_models = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model_dist.train()
        model_head.train()
        total_loss = 0

        for batch in train_loader:
            # Distance Model eğitimi
            opt_dist.zero_grad()
            features = batch['features'].to(device)
            dist_target = batch['dist'].to(device)
            dist_output = model_dist(features)
            loss_dist = criterion(dist_output, dist_target)
            loss_dist.backward()
            torch.nn.utils.clip_grad_norm_(model_dist.parameters(), 1.0)
            opt_dist.step()

            # Heading Model eğitimi
            opt_head.zero_grad()
            head_target = batch['heading'].to(device)
            head_output = model_head(features)
            loss_head = criterion(head_output, head_target)
            loss_head.backward()
            torch.nn.utils.clip_grad_norm_(model_head.parameters(), 1.0)
            opt_head.step()

            total_loss += (loss_dist.item() + loss_head.item())

        val_mse_dist, _ = evaluate(model_dist, val_loader, 'dist', dist_scaler, device)
        val_mse_head, _ = evaluate(model_head, val_loader, 'heading', head_scaler, device)
        avg_val_mse = (val_mse_dist + val_mse_head) / 2

        if avg_val_mse < best_val_mse - 0.001:
            best_val_mse = avg_val_mse
            best_models = (
                copy.deepcopy(model_dist.state_dict()),
                copy.deepcopy(model_head.state_dict())
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                break

    return best_val_mse, best_models

def evaluate(model, loader, target_type, scaler, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            target = batch[target_type].cpu().numpy()
            outputs = model(features).cpu().numpy()
            preds.extend(scaler.inverse_transform(outputs).flatten())
            targets.extend(scaler.inverse_transform(target.reshape(-1, 1)).flatten())
    return mean_squared_error(targets, preds), mean_absolute_error(targets, preds)

# =============================================================================
# Ana Eğitim Fonksiyonu
# =============================================================================
def train_models(data_dir):
    # Outlier filtering uygulanarak veriler yükleniyor
    X_raw, dist_raw, head_raw, outlier_scaler, outlier_params = load_data_from_folder(data_dir, filter_outliers=True, threshold_percentile=99)
    feat_cols = [f"feat_{i}" for i in range(X_raw.shape[1])]
    df = pd.DataFrame(X_raw, columns=feat_cols)
    df["dist"] = dist_raw
    df["heading"] = head_raw

    if ADD_SYNTHETIC:
        syn_df = create_synthetic_data(df, feat_cols, SYNTH_SAMPLES)
        df = pd.concat([df, syn_df], ignore_index=True)

    if USE_AUGMENT:
        aug_df = augment_data(df, feat_cols, AUG_TIMES)
        df = pd.concat([df, aug_df], ignore_index=True)

    X_feat = df[feat_cols].values
    y_dist = df["dist"].values.reshape(-1, 1)
    y_head = df["heading"].values.reshape(-1, 1)

    # Son aşamada kullanılan scaler’lar
    feat_scaler = QuantileTransformer(n_quantiles=500, output_distribution='normal')
    dist_scaler = QuantileTransformer(n_quantiles=500, output_distribution='normal')
    head_scaler = QuantileTransformer(n_quantiles=500, output_distribution='normal')

    X_scaled = feat_scaler.fit_transform(X_feat)
    y_dist_scaled = dist_scaler.fit_transform(y_dist)
    y_head_scaled = head_scaler.fit_transform(y_head)

    X_train, X_test, y_dist_train, y_dist_test, y_head_train, y_head_test = train_test_split(
        X_scaled, y_dist_scaled, y_head_scaled, test_size=0.2, random_state=42
    )
    X_train, X_val, y_dist_train, y_dist_val, y_head_train, y_head_val = train_test_split(
        X_train, y_dist_train, y_head_train, test_size=0.1, random_state=42
    )

    param_combinations = [dict(zip(PARAM_GRID.keys(), values))
                          for values in itertools.product(*PARAM_GRID.values())]

    best_score = float('inf')
    best_params = None
    best_models = None

    print(f"Starting GridSearch with {len(param_combinations)} combinations...")

    for i, params in enumerate(param_combinations):
        start_time = time()
        print(f"\nCombination {i + 1}/{len(param_combinations)}")
        print("Parameters:", params)

        val_score, models = train_model(
            params=params,
            X_train=X_train,
            y_dist_train=y_dist_train.flatten(),
            y_head_train=y_head_train.flatten(),
            X_val=X_val,
            y_dist_val=y_dist_val.flatten(),
            y_head_val=y_head_val.flatten(),
            feat_scaler=feat_scaler,
            dist_scaler=dist_scaler,
            head_scaler=head_scaler,
            input_dim=X_train.shape[1]
        )

        if val_score < best_score:
            best_score = val_score
            best_params = params
            best_models = models
            print(f"New best score: {best_score:.4f}")

        print(f"Elapsed time: {time() - start_time:.1f}s")

    print("\nBest parameters:", best_params)
    print(f"Best validation score: {best_score:.4f}")

    save_models_directory = "results/LSTM-model/"
    os.makedirs(save_models_directory, exist_ok=True)
    torch.save(best_models[0], os.path.join(save_models_directory, "best_dist_model.pth"))
    torch.save(best_models[1], os.path.join(save_models_directory, "best_heading_model.pth"))
    joblib.dump(feat_scaler, os.path.join(save_models_directory, "best_feat_scaler.pkl"))
    joblib.dump(dist_scaler, os.path.join(save_models_directory, "best_dist_scaler.pkl"))
    joblib.dump(head_scaler, os.path.join(save_models_directory, "best_heading_scaler.pkl"))
    joblib.dump(feat_cols, os.path.join(save_models_directory, "best_feat_cols.pkl"))
    joblib.dump(best_params, os.path.join(save_models_directory, "best_params.pkl"))
    if outlier_scaler is not None and outlier_params is not None:
        joblib.dump(outlier_scaler, os.path.join(save_models_directory, "best_outlier_scaler.pkl"))
        joblib.dump(outlier_params, os.path.join(save_models_directory, "best_outlier_params.pkl"))
    print("\nTraining is completed. Optimal models and outlier filtering parameters are saved.")

if __name__ == "__main__":
    data_directory = "data/training"  # Eğitim verilerinizin bulunduğu dizin
    train_models(data_directory)
