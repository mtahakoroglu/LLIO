import os
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.stats import skew, kurtosis, iqr
from scipy.signal import welch
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

##############################################################################
# PARAMETRELER
##############################################################################
FS = 200                # IMU örnekleme hızı
ADD_SYNTHETIC = True    # Sentetik veri eklensin mi?
SYNTH_SAMPLES = 800    # Kaç sentetik örnek üretilecek?
USE_AUGMENT = True      # Data augmentation
AUG_TIMES = 1       # Kaç kat augment
N_FOLDS = 8             # Cross-validation fold sayısı

GB_MODEL_PATH = "results/GB-model"
DIST_MODEL_PATH = GB_MODEL_PATH + "/" + "dist_model.pkl"
HEADING_MODEL_PATH = GB_MODEL_PATH + "/" + "heading_model.pkl"
DIST_SCALER_PATH = GB_MODEL_PATH + "/" + "dist_scaler.pkl"
HEADING_SCALER_PATH = GB_MODEL_PATH + "/" + "heading_scaler.pkl"
FEATCOLS_PATH = GB_MODEL_PATH + "/" + "feat_cols.pkl"

##############################################################################
# 1) Özellik Çıkarma (3 eksen ivme)
##############################################################################

def extract_time_domain_features(stride_data):
    """
    stride_data: (T,3) => ax, ay, az
    """
    feats = []
    n_ch = stride_data.shape[1]
    for ch in range(n_ch):
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
        feats.append(np.sum(Pxx))     # total power
        feats.append(np.mean(Pxx))    # mean power
        feats.append(np.max(Pxx))     # peak power
        feats.append(np.median(Pxx))  # median power
    return feats

def extract_autocorr_features(stride_data, lags=5):
    feats = []
    n_ch = stride_data.shape[1]
    for ch in range(n_ch):
        sig = stride_data[:, ch]
        sig_centered = sig - np.mean(sig)
        ac = np.correlate(sig_centered, sig_centered, mode='full')
        ac = ac[len(ac)//2:]
        if ac[0] != 0:
            ac = ac / ac[0]
        for lag in range(1, lags+1):
            if lag < len(ac):
                feats.append(ac[lag])
            else:
                feats.append(0)
    return feats

def extract_stride_features(stride_data, fs=200):
    """
    Genişletilmiş özellik seti (time + freq + autocorr + peakAcc + strideTime).
    stride_data: (T,3)
    """
    feats_time = extract_time_domain_features(stride_data)
    feats_freq = extract_frequency_domain_features(stride_data, fs)
    feats_ac   = extract_autocorr_features(stride_data, lags=5)

    acc_norm = np.linalg.norm(stride_data, axis=1)
    peakAcc  = np.max(acc_norm) if len(acc_norm)>0 else 0
    dt = 1.0/fs
    stride_time = stride_data.shape[0]*dt

    return feats_time + feats_freq + feats_ac + [peakAcc, stride_time]

##############################################################################
# 2) Sentetik Veri ve Augmentation
##############################################################################

def create_synthetic_data(df, feat_cols, n_samples=300):
    """
    - Rastgele satırlardan gürültü eklenmiş kopyalar üretir
    - dist/heading'de küçük sapma
    """
    rows = []
    for _ in range(n_samples):
        row = df.sample(n=1).iloc[0].copy()
        for col in feat_cols:
            noise_scale = 0.01*abs(row[col])
            noise = np.random.normal(0, noise_scale)
            row[col] += noise
        # dist
        row["dist"] += np.random.normal(0, 0.03*row["dist"])
        row["heading"] += np.random.normal(0, 0.05)
        # heading normalle
        row["heading"] = (row["heading"] + np.pi) % (2*np.pi) - np.pi
        rows.append(row)
    return pd.DataFrame(rows, columns=df.columns)

def augment_data(df, feat_cols, times=1):
    """
    - Tüm satırları çoğaltır
    - Ek gürültü ve ölçek uygulaması
    """
    aug_rows = []
    for _ in range(times):
        for idx in range(len(df)):
            row = df.iloc[idx].copy()
            for col in feat_cols:
                noise_scale = 0.01*abs(row[col])
                noise = np.random.normal(0, noise_scale)
                row[col] += noise
                scale_factor = np.random.uniform(0.95, 1.05)
                row[col] *= scale_factor
            row["dist"] *= np.random.uniform(0.95, 1.05)
            row["heading"] += np.random.normal(0, 0.02)
            row["heading"] = (row["heading"] + np.pi) % (2*np.pi) - np.pi
            aug_rows.append(row)
    return pd.DataFrame(aug_rows, columns=df.columns)

##############################################################################
# 3) .mat dosyasını parse => X, dist, heading
##############################################################################

def parse_mat_file(file_path):
    data = loadmat(file_path)
    stride_idx = data["strideIndex"].flatten()
    GCP = data["GCP"]       # (S,2)
    acc_n = data["acc_n"]   # (N,3)

    n_strides = min(len(stride_idx)-1, GCP.shape[0])
    X_list = []
    dist_list = []
    head_list = []

    for i in range(n_strides):
        s = stride_idx[i]
        e = stride_idx[i+1]
        seg = acc_n[s:e, :]
        if seg.shape[0] < 2:
            continue
        feats = extract_stride_features(seg, fs=FS)

        if i < n_strides - 1:
            dx = GCP[i+1,0] - GCP[i,0]
            dy = GCP[i+1,1] - GCP[i,1]
        else:
            dx, dy = 0, 0

        dist_ = np.sqrt(dx**2 + dy**2)
        heading_ = np.arctan2(dy, dx)
        X_list.append(feats)
        dist_list.append(dist_)
        head_list.append(heading_)

    return np.array(X_list), np.array(dist_list), np.array(head_list)

def load_data_from_folder(data_dir):
    all_feats = []
    all_dist  = []
    all_head  = []

    for fname in os.listdir(data_dir):
        if fname.endswith(".mat"):
            fpath = os.path.join(data_dir, fname)
            Xf, Df, Hf = parse_mat_file(fpath)
            if Xf.shape[0] > 0:
                all_feats.append(Xf)
                all_dist.append(Df)
                all_head.append(Hf)

    if len(all_feats) == 0:
        raise ValueError("Klasörde geçerli .mat dosyası bulunamadı.")

    X = np.vstack(all_feats)
    Dist = np.hstack(all_dist)
    Head = np.hstack(all_head)
    return X, Dist, Head

##############################################################################
# 4) TRAIN MODELS (ayrı dist ve heading)
##############################################################################

def train_models(data_dir):
    # 1) Veriyi al
    X_raw, dist_raw, head_raw = load_data_from_folder(data_dir)

    # 2) Feature isimleri (59 adet)
    feat_cols = []
    for ch in range(3):
        feat_cols += [
            f"ch{ch}_mean", f"ch{ch}_std", f"ch{ch}_max", f"ch{ch}_min",
            f"ch{ch}_skew", f"ch{ch}_kurt", f"ch{ch}_iqr", f"ch{ch}_median",
            f"ch{ch}_energy", f"ch{ch}_var"
        ]
    for ch in range(3):
        feat_cols += [
            f"ch{ch}_totPow", f"ch{ch}_meanPow", f"ch{ch}_peakPow", f"ch{ch}_medianPow"
        ]
    for ch in range(3):
        for lag in range(1,6):
            feat_cols.append(f"ch{ch}_acLag{lag}")
    feat_cols += ["peakAcc","stride_time"]

    df = pd.DataFrame(X_raw, columns=feat_cols)
    df["dist"] = dist_raw
    df["heading"] = head_raw

    # 3) Sentetik veri (opsiyonel)
    if ADD_SYNTHETIC:
        syn_df = create_synthetic_data(df, feat_cols, n_samples=SYNTH_SAMPLES)
        df = pd.concat([df, syn_df], ignore_index=True)

    # 4) Augmentation (opsiyonel)
    if USE_AUGMENT and AUG_TIMES>0:
        aug_df = augment_data(df, feat_cols, times=AUG_TIMES)
        df = pd.concat([df, aug_df], ignore_index=True)

    print("Toplam stride sayısı:", len(df))

    # 5) Ayrı modeller için X / Y
    # Dist Model
    X_feat = df[feat_cols].values
    y_dist = df["dist"].values

    # Heading Model
    y_head = df["heading"].values

    #----------------------------------------------------------------------
    # DISTANCE MODEL
    #----------------------------------------------------------------------
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_feat, y_dist, test_size=0.2, random_state=42
    )
    dist_scaler = StandardScaler()
    X_train_d_sc = dist_scaler.fit_transform(X_train_d)
    X_test_d_sc  = dist_scaler.transform(X_test_d)

    param_grid_dist = {
        "n_estimators": [100, 200, 800],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0]
    }
    dist_model = XGBRegressor(random_state=42, objective="reg:squarederror")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    gs_dist = GridSearchCV(dist_model, param_grid_dist,
                           cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    gs_dist.fit(X_train_d_sc, y_train_d)
    best_dist_model = gs_dist.best_estimator_
    print("En iyi Dist Model param:", gs_dist.best_params_)

    # Test skoru
    pred_dist_test = best_dist_model.predict(X_test_d_sc)
    mse_dist = mean_squared_error(y_test_d, pred_dist_test)
    print(f"Distance Model Test MSE: {mse_dist:.4f}")

    #----------------------------------------------------------------------
    # HEADING MODEL
    #----------------------------------------------------------------------
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_feat, y_head, test_size=0.2, random_state=42
    )
    heading_scaler = StandardScaler()
    X_train_h_sc = heading_scaler.fit_transform(X_train_h)
    X_test_h_sc  = heading_scaler.transform(X_test_h)

    param_grid_head = {
        "n_estimators": [100, 200, 800],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0]
    }
    head_model = XGBRegressor(random_state=42, objective="reg:squarederror")
    gs_head = GridSearchCV(head_model, param_grid_head,
                           cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    gs_head.fit(X_train_h_sc, y_train_h)
    best_head_model = gs_head.best_estimator_
    print("En iyi Heading Model param:", gs_head.best_params_)

    # Test skoru
    pred_head_test = best_head_model.predict(X_test_h_sc)
    mse_head = mean_squared_error(y_test_h, pred_head_test)
    print(f"Heading Model Test MSE: {mse_head:.4f}")

    #----------------------------------------------------------------------
    # Modelleri Kaydet
    #----------------------------------------------------------------------
    joblib.dump(best_dist_model, DIST_MODEL_PATH)
    joblib.dump(best_head_model, HEADING_MODEL_PATH)
    joblib.dump(dist_scaler, DIST_SCALER_PATH)
    joblib.dump(heading_scaler, HEADING_SCALER_PATH)
    joblib.dump(feat_cols, FEATCOLS_PATH)

    print("dist_model, heading_model, dist_scaler, heading_scaler, feat_cols kaydedildi.")

if __name__ == "__main__":
    data_directory = r"data\LLIO_training_data"
    train_models(data_directory)
