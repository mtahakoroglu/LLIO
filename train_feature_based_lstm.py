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

##############################################################################
# GÜNCELLENMİŞ PARAMETRELER
##############################################################################
FS = 200
ADD_SYNTHETIC = True
SYNTH_SAMPLES = 1200  # Artırılmış sentetik veri
USE_AUGMENT = True
AUG_TIMES = 4 # 2x augmentasyon


##############################################################################
# 1) Özellik Çıkarma (Değişmedi)
##############################################################################
# ... (extract_stride_features ve diğer fonksiyonlar aynı kalacak) ...

##############################################################################
# 2) Gelişmiş Model Mimarisi
##############################################################################
class IMUDataset(Dataset):
    def __init__(self, X, y_dist, y_head, feat_scaler=None, dist_scaler=None, head_scaler=None):
        # Özellikleri ölçekle
        if feat_scaler:
            self.X = feat_scaler.transform(X.astype(np.float32))
        else:
            self.X = X.astype(np.float32)

        # Hedefleri ölçekle
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

def parse_mat_file(file_path):
    data = loadmat(file_path)
    stride_idx = data["strideIndex"].flatten()
    GCP = data["GCP"]
    acc_n = data["acc_n"]

    X_list, dist_list, head_list = [], [], []

    for i in range(len(stride_idx) - 1):
        s, e = stride_idx[i], stride_idx[i + 1]
        seg = acc_n[s:e, :]
        if seg.shape[0] < 10:  # Minimum 10 örnek kontrolü
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

def load_data_from_folder(data_dir):
    all_feats, all_dist, all_head = [], [], []

    for fname in os.listdir(data_dir):
        if fname.endswith(".mat"):
            Xf, Df, Hf = parse_mat_file(os.path.join(data_dir, fname))
            if len(Xf) > 0:
                all_feats.append(Xf)
                all_dist.append(Df)
                all_head.append(Hf)

    X = np.vstack(all_feats)
    Dist = np.hstack(all_dist)
    Head = np.hstack(all_head)
    X = np.nan_to_num(X)
    X[np.isposinf(X)] = 0
    X[np.isneginf(X)] = 0

    return X, Dist, Head
def create_synthetic_data(df, feat_cols, n_samples=300):
    rows = []
    for _ in range(n_samples):
        row = df.sample(n=1).iloc[0].copy()
        # Özellikler için gürültü ekle
        for col in feat_cols:
            noise_scale = 0.02 * abs(row[col])
            row[col] += np.random.normal(0, noise_scale)

        # Hedefler için varyasyon
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
            # Özellik augmentasyonu
            for col in feat_cols:
                row[col] *= np.random.uniform(0.98, 1.02)
                row[col] += np.random.normal(0, 0.01 * abs(row[col]))

            # Hedef augmentasyonu
            row["dist"] *= np.random.uniform(0.99, 1.01)
            row["heading"] += np.random.uniform(-0.005, 0.005)
            row["heading"] = (row["heading"] + np.pi) % (2 * np.pi) - np.pi

            aug_rows.append(row)
    return pd.DataFrame(aug_rows)
##############################################################################
# 3) İyileştirilmiş Eğitim Pipeline'ı
##############################################################################
def train_models(data_dir):
    # Veri yükleme ve ön işleme (aynı)
    X_raw, dist_raw, head_raw = load_data_from_folder(data_dir)
    feat_cols = [f"feat_{i}" for i in range(X_raw.shape[1])]
    df = pd.DataFrame(X_raw, columns=feat_cols)
    df["dist"] = dist_raw
    df["heading"] = head_raw

    # Sentetik veri ve augmentasyon
    if ADD_SYNTHETIC:
        syn_df = create_synthetic_data(df, feat_cols, SYNTH_SAMPLES)
        df = pd.concat([df, syn_df], ignore_index=True)

    if USE_AUGMENT:
        aug_df = augment_data(df, feat_cols, AUG_TIMES)
        df = pd.concat([df, aug_df], ignore_index=True)

    # Veri normalizasyonu
    X_feat = df[feat_cols].values
    y_dist = df["dist"].values.reshape(-1, 1)
    y_head = df["heading"].values.reshape(-1, 1)

    feat_scaler = QuantileTransformer(n_quantiles=500, output_distribution='normal')
    dist_scaler = QuantileTransformer(n_quantiles=500, output_distribution='normal')
    head_scaler = QuantileTransformer(n_quantiles=500, output_distribution='normal')

    X_scaled = feat_scaler.fit_transform(X_feat)
    y_dist_scaled = dist_scaler.fit_transform(y_dist)
    y_head_scaled = head_scaler.fit_transform(y_head)

    # Veri bölümleme
    X_train, X_test, y_dist_train, y_dist_test, y_head_train, y_head_test = train_test_split(
        X_scaled, y_dist_scaled, y_head_scaled, test_size=0.0001, random_state=42
    )
    X_train, X_val, y_dist_train, y_dist_val, y_head_train, y_head_val = train_test_split(
        X_train, y_dist_train, y_head_train, test_size=0.1, random_state=42
    )

    # DataLoader'lar
    train_dataset = IMUDataset(X_train, y_dist_train.flatten(), y_head_train.flatten())
    val_dataset = IMUDataset(X_val, y_dist_val.flatten(), y_head_val.flatten())
    test_dataset = IMUDataset(X_test, y_dist_test.flatten(), y_head_test.flatten())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model ve Optimizasyon
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dist = IMUModel(X_train.shape[1]).to(device)
    model_head = IMUModel(X_train.shape[1]).to(device)

    criterion = nn.HuberLoss(delta=0.5)
    opt_dist = optim.AdamW(model_dist.parameters(), lr=0.001, weight_decay=1e-4)
    opt_head = optim.AdamW(model_head.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler_dist = optim.lr_scheduler.ReduceLROnPlateau(opt_dist, 'min', patience=15, factor=0.5, verbose=True)
    scheduler_head = optim.lr_scheduler.ReduceLROnPlateau(opt_head, 'min', patience=15, factor=0.5, verbose=True)

    # Eğitim Fonksiyonları
    def evaluate(model, loader, target_type, scaler):
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

    # Eğitim Döngüsü
    best_dist_model, best_head_model = None, None
    best_val_mse = float('inf')
    early_stop_patience = 30
    epochs_no_improve = 0

    for epoch in range(500):
        # Training Phase
        model_dist.train()
        model_head.train()
        total_loss = 0

        for batch in train_loader:
            # Distance Model Update
            opt_dist.zero_grad()
            features = batch['features'].to(device)
            dist_target = batch['dist'].to(device)

            dist_output = model_dist(features)
            loss_dist = criterion(dist_output, dist_target)
            loss_dist.backward()
            torch.nn.utils.clip_grad_norm_(model_dist.parameters(), 1.0)
            opt_dist.step()

            # Heading Model Update
            opt_head.zero_grad()
            head_target = batch['heading'].to(device)

            head_output = model_head(features)
            loss_head = criterion(head_output, head_target)
            loss_head.backward()
            torch.nn.utils.clip_grad_norm_(model_head.parameters(), 1.0)
            opt_head.step()

            total_loss += (loss_dist.item() + loss_head.item())

        # Validation Phase
        val_mse_dist, val_mae_dist = evaluate(model_dist, val_loader, 'dist', dist_scaler)
        val_mse_head, val_mae_head = evaluate(model_head, val_loader, 'heading', head_scaler)
        avg_val_mse = (val_mse_dist + val_mse_head) / 2

        # Learning Rate Update
        scheduler_dist.step(val_mse_dist)
        scheduler_head.step(val_mse_head)

        # Early Stopping Kontrolü
        if avg_val_mse < best_val_mse - 0.001:
            best_val_mse = avg_val_mse
            best_dist_model = copy.deepcopy(model_dist.state_dict())
            best_head_model = copy.deepcopy(model_head.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Progress Reporting
        test_mse_dist, test_mae_dist = evaluate(model_dist, test_loader, 'dist', dist_scaler)
        test_mse_head, test_mae_head = evaluate(model_head, test_loader, 'heading', head_scaler)

        print(f"\nEpoch {epoch + 1}")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Validation MSE - Dist: {val_mse_dist:.4f} | Head: {val_mse_head:.4f}")
        print(f"Test MSE - Dist: {test_mse_dist:.4f} | Head: {test_mse_head:.4f}")
        print(f"Test MAE - Dist: {test_mae_dist:.4f} | Head: {test_mae_head:.4f}")

        if epochs_no_improve >= early_stop_patience:
            print("\nEarly stopping triggered!")
            break

    # Final Model Kaydetme
    save_models_directory = "results/LSTM-model/"
    torch.save(best_dist_model, f"{save_models_directory}improved_dist_model.pth")
    torch.save(best_head_model, f"{save_models_directory}improved_heading_model.pth")
    joblib.dump(feat_scaler, f"{save_models_directory}feat_scaler.pkl")
    joblib.dump(dist_scaler, f"{save_models_directory}dist_scaler.pkl")
    joblib.dump(head_scaler, f"{save_models_directory}heading_scaler.pkl")
    joblib.dump(feat_cols, f"{save_models_directory}feat_cols.pkl")

    print("\nTraining completed. Models saved.")


if __name__ == "__main__":
    data_directory = "data/training"
    train_models(data_directory)