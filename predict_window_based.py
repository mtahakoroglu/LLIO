# pred_no_gyro_styled_blocking_saved_plots.py

import os
from glob import glob
import joblib
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch

# ======================
# --- AYARLAR & SABİTLER ---
# ======================
MAX_LEN       = 256
PRED_DIR      = "data/testing"   # .mat dosyalarının olduğu dizin
PRETRAINED_MODELS_DIR = "results/window-based"
X_SCALER_PATH = os.path.join(PRETRAINED_MODELS_DIR, "x_scaler.pkl")
Y_SCALER_PATH = os.path.join(PRETRAINED_MODELS_DIR, "y_scaler.pkl")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# --- PLOT KAYDETME AYARLARI ---
# ======================
# Mevcut "results/window-based" klasörüne grafikleri kaydedeceğiz.
PLOT_DIR = os.path.join(".", "results", "figs", "window-based")

# ===================================
# --- SCALER DOSYALARINI YÜKLEME ---
# ===================================
if not os.path.isfile(X_SCALER_PATH) or not os.path.isfile(Y_SCALER_PATH):
    raise FileNotFoundError(f"Scaler dosyaları bulunamadı:\n  {X_SCALER_PATH}\n  {Y_SCALER_PATH}")

x_scaler = joblib.load(X_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

# x_scaler özellik sayısını kontrol et (acc_x, acc_y, acc_z, src_flag = 4 olmalı)
if hasattr(x_scaler, 'n_features_in_'):
    n_in = x_scaler.n_features_in_
    if n_in != 4:
        raise RuntimeError(f"x_scaler.n_features_in_ = {n_in}. Bu kod yalnızca 4-feature için hazırlanmıştır.")
    print(f"[Bilgi] x_scaler, {n_in} özelliğe (acc(3)+src_flag) göre eğitilmiş.\n")
else:
    print("[Bilgi] x_scaler yüklendi. Feature sayısı kontrolü yapılamadı (eski scikit-learn sürümü).\n")

# =================================================
# --- MODEL DOSYALARINI BULMA & YÜKLEME ---
# =================================================
def find_model_file(pattern: str):
    """
    Verilen pattern ile eşleşen .pt dosyalarını glob ile bul.
    Birden fazla eşleşme varsa sıralı ilkini döndür.
    Eğer yoksa FileNotFoundError fırlat.
    """
    files = sorted(glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"Model dosyası bulunamadı: {pattern}")
    return files[0]

def rotate_trajectory(trajectory, theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return trajectory @ rotation_matrix.T

transformer_path = find_model_file(f"{PRETRAINED_MODELS_DIR}/best_model_transformer_*.pt")
lstm_path        = find_model_file(f"{PRETRAINED_MODELS_DIR}/best_model_lstm_*.pt")
cnn_path         = find_model_file(f"{PRETRAINED_MODELS_DIR}/best_model_cnn_*.pt")

print("Yüklenen modeller:")
print(f"  Transformer: {os.path.basename(transformer_path)}")
print(f"  LSTM:        {os.path.basename(lstm_path)}")
print(f"  CNN:         {os.path.basename(cnn_path)}\n")

model_transformer = torch.jit.load(transformer_path, map_location=DEVICE).to(DEVICE).eval()
model_lstm        = torch.jit.load(lstm_path,        map_location=DEVICE).to(DEVICE).eval()
model_cnn         = torch.jit.load(cnn_path,         map_location=DEVICE).to(DEVICE).eval()

# ===================================
# --- TEST KLASÖRÜNDEKİ .mat DOSYALARINI DOLAŞ ---
# ===================================
mat_files = sorted(glob(os.path.join(PRED_DIR, "*.mat")))
if len(mat_files) == 0:
    print(f"[UYARI] '{PRED_DIR}' dizininde .mat dosyası bulunamadı.")
    exit(0)

# ====================================
# --- HER .mat DOSYASI İÇİN DÖNGÜ ---
# ====================================
for mat_fp in mat_files:
    basename = os.path.splitext(os.path.basename(mat_fp))[0]
    try:
        data = loadmat(mat_fp)
    except Exception as e:
        print(f"[HATA] '{mat_fp}' yüklenirken hata:\n  {e}")
        continue

    # Gerekli anahtarlar var mı?
    if "strideIndex" not in data or "acc_n" not in data or "GCP" not in data:
        print(f"[UYARI] '{basename}.mat' içinde strideIndex, acc_n veya GCP yok. Atlanıyor.")
        continue

    expID, idx, GCP, acc = data["expID"].item(), data["strideIndex"].flatten().astype(int), data["GCP_wcf"], data["acc_n"]
    theta = data["theta"].flatten()[0]; pyshoe_trajectory = data["pyshoeTrajectory"]
    pyshoe_trajectory_wcf = np.squeeze(rotate_trajectory(pyshoe_trajectory, -theta)) # ncf to wcf
    pyshoe_trajectory_wcf[:,1] = -pyshoe_trajectory_wcf[:,1] # change made by mtahakoroglu to match with GT alignment
    # take only stride_stride_idx strides from pyshoe_trajectory_wcf
    pyshoe_trajectory_wcf = pyshoe_trajectory_wcf[idx, :]
    src_flag = 1 if "SensorConnectData" in basename else 0

    # True ve tahmin dx, dy listeleri
    true_dx_list = []
    true_dy_list = []
    pred_trans_dx_list = []
    pred_trans_dy_list = []
    pred_lstm_dx_list  = []
    pred_lstm_dy_list  = []
    pred_cnn_dx_list   = []
    pred_cnn_dy_list   = []

    # --------- Her stride segmentini işle ---------
    for i in range(len(idx) - 1):
        s, e = idx[i], idx[i + 1]
        if s < 0 or e > acc.shape[0] or e <= s:
            continue

        seg_a = acc[s:e, :]   # (segment_length × 3)
        if seg_a.shape[0] < 10:
            continue

        # Ground Truth dx, dy
        dx_true = float(GCP[i + 1, 0] - GCP[i, 0])
        dy_true = float(GCP[i + 1, 1] - GCP[i, 1])
        true_dx_list.append(dx_true)
        true_dy_list.append(dy_true)

        # ------------------------
        # Tahmin için girdi hazırla
        # ------------------------
        def pad_truncate(seg: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
            L = seg.shape[0]
            if L >= max_len:
                return seg[:max_len]
            pad = np.tile(seg[-1:], (max_len - L, 1))
            return np.vstack([seg, pad])

        a_p = pad_truncate(seg_a)  # (MAX_LEN, 3)
        f_p = np.full((MAX_LEN, 1), src_flag, dtype=np.float32)  # (MAX_LEN, 1)

        # X_np: (MAX_LEN, 4)
        X_np = np.concatenate([a_p.astype(np.float32), f_p], axis=1)

        # Ölçekle
        X_flat = X_np.reshape(-1, 4)  # (MAX_LEN, 4)
        try:
            Xs = x_scaler.transform(X_flat)  # (MAX_LEN, 4)
        except ValueError as e:
            raise RuntimeError(f"[HATA] Scaler.transform hatası: {e}")

        Xs = Xs.reshape(1, MAX_LEN, 4).astype(np.float32)
        X_tensor = torch.from_numpy(Xs).to(DEVICE)

        # ------------------------
        # MODEL TAHMİNLERİ
        # ------------------------
        with torch.no_grad():
            # 1) Transformer
            y_norm_trans = model_transformer(X_tensor).cpu().numpy().reshape(1, 2)
            dx_pred_trans, dy_pred_trans = y_scaler.inverse_transform(y_norm_trans)[0]
            pred_trans_dx_list.append(float(dx_pred_trans))
            pred_trans_dy_list.append(float(dy_pred_trans))

            # 2) LSTM
            y_norm_lstm = model_lstm(X_tensor).cpu().numpy().reshape(1, 2)
            dx_pred_lstm, dy_pred_lstm = y_scaler.inverse_transform(y_norm_lstm)[0]
            pred_lstm_dx_list.append(float(dx_pred_lstm))
            pred_lstm_dy_list.append(float(dy_pred_lstm))

            # 3) CNN
            y_norm_cnn = model_cnn(X_tensor).cpu().numpy().reshape(1, 2)
            dx_pred_cnn, dy_pred_cnn = y_scaler.inverse_transform(y_norm_cnn)[0]
            pred_cnn_dx_list.append(float(dx_pred_cnn))
            pred_cnn_dy_list.append(float(dy_pred_cnn))

    # Eğer geçerli stride yoksa atla
    if len(true_dx_list) == 0:
        print(f"[BILGI] '{basename}' dosyasında geçerli stride bulunamadı. Atlanıyor.")
        continue

    # ============================================
    # --- GERÇEK ve TAHMİN YÜRÜYÜŞÜ HESAPLA & ÇİZ ---
    # ============================================
    x_true = [float(GCP[0, 0])]
    y_true = [float(GCP[0, 1])]

    x_trans = [float(GCP[0, 0])]
    y_trans = [float(GCP[0, 1])]

    x_lstm  = [float(GCP[0, 0])]
    y_lstm  = [float(GCP[0, 1])]

    x_cnn   = [float(GCP[0, 0])]
    y_cnn   = [float(GCP[0, 1])]

    for dx_t, dy_t, dx_p_t, dy_p_t, dx_p_l, dy_p_l, dx_p_c, dy_p_c in zip(
            true_dx_list, true_dy_list,
            pred_trans_dx_list, pred_trans_dy_list,
            pred_lstm_dx_list, pred_lstm_dy_list,
            pred_cnn_dx_list,  pred_cnn_dy_list):
        # Ground Truth ekle
        x_true.append(x_true[-1] + dx_t)
        y_true.append(y_true[-1] + dy_t)
        # Transformer ekle
        x_trans.append(x_trans[-1] + dx_p_t)
        y_trans.append(y_trans[-1] + dy_p_t)
        # LSTM ekle
        x_lstm.append(x_lstm[-1] + dx_p_l)
        y_lstm.append(y_lstm[-1] + dy_p_l)
        # CNN ekle
        x_cnn.append(x_cnn[-1] + dx_p_c)
        y_cnn.append(y_cnn[-1] + dy_p_c)
    ################ DR GOKHAN CETIN ALREADY COMPUTED CUMULATIVE VARIABLES SO SKIPPIN ADDING OPERATION HERE #############    
    # x_pred_lstm = np.concatenate([[0.], np.cumsum(x_lstm)]); y_pred_lstm = np.concatenate([[0.], np.cumsum(y_lstm)])
    # x_pred_cnn = np.concatenate([[0.], np.cumsum(x_cnn)]); y_pred_cnn = np.concatenate([[0.], np.cumsum(y_cnn)])
    # x_pred_trans = np.concatenate([[0.], np.cumsum(x_trans)]); y_pred_trans = np.concatenate([[0.], np.cumsum(y_trans)])
    # now change of basis from ncf to wcf
    prediction_lstm = np.column_stack((x_lstm, y_lstm))
    prediction_cnn = np.column_stack((x_cnn, y_cnn))
    prediction_trans = np.column_stack((x_trans, y_trans))
    prediction_lstm_wcf = np.squeeze(rotate_trajectory(prediction_lstm, -theta))
    prediction_cnn_wcf = np.squeeze(rotate_trajectory(prediction_cnn, -theta))
    prediction_trans_wcf = np.squeeze(rotate_trajectory(prediction_trans, -theta))
    prediction_lstm_wcf[:,1] = -prediction_lstm_wcf[:,1] # change made by mtahakoroglu to match with GT alignment
    prediction_cnn_wcf[:,1] = -prediction_cnn_wcf[:,1] # change made by mtahakoroglu to match with GT alignment
    prediction_trans_wcf[:,1] = -prediction_trans_wcf[:,1] # change made by mtahakoroglu to match with GT alignment
    # ================
    # --- Grafikleri çizdirme ve kaydetme ---
    # ================
    # 1) Window-based LSTM data-driven INS grafiği
    plt.figure(figsize=(10, 6))
    plt.scatter(GCP[:,0], GCP[:,1], c='orange', marker='s', s=50, edgecolors='k', label='GCP')
    plt.plot(GCP[:,0], GCP[:,1], 'orange', linestyle='--', linewidth=1.5)
    plt.plot(prediction_lstm_wcf[:,0], prediction_lstm_wcf[:,1], 'b.-', linewidth=2, label='Window-based LSTM')
    # if expID not in [1, 51, 52, 53, 54]:
    if expID not in [51, 53, 54, 65, 66, 68, 75]:
        plt.plot(pyshoe_trajectory_wcf[:,0], pyshoe_trajectory_wcf[:,1], 'r.-', linewidth=1.5, label='PyShoe (LSTM)')
    plt.title(f"{basename.split('_LLIO')[0]}", fontsize=16)
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
    # lstm_plot_path = os.path.join(PLOT_DIR, f"{basename.split('_LLIO.mat')[0]}_window_based_lstm.png")
    plt.savefig(f"results/figs/window-based/window_based_lstm_exp_{expID}.png", dpi=300)
    # plt.savefig(lstm_plot_path, dpi=300)
    # plt.savefig(f"results/figs/hybrid/hybrid_exp_{expID}.png", dpi=300)
    plt.pause(1)  # Show the figure for 1 second
    plt.close()   # Close the figure automatically

    # 2) Window-based CNN data-driven INS grafiği
    plt.figure(figsize=(10, 6))
    plt.scatter(GCP[:,0], GCP[:,1], c='orange', marker='s', s=50, edgecolors='k', label='GCP')
    plt.plot(GCP[:,0], GCP[:,1], 'orange', linestyle='--', linewidth=1.5)
    plt.plot(prediction_cnn_wcf[:,0], prediction_cnn_wcf[:,1], 'b.-', linewidth=2, label='Window-based CNN')
    # if expID not in [1, 51, 52, 53, 54]:
    if expID not in [51, 53, 54, 65, 66, 68, 75]:
        plt.plot(pyshoe_trajectory_wcf[:,0], pyshoe_trajectory_wcf[:,1], 'r.-', linewidth=1.5, label='PyShoe (LSTM)')
    plt.title(f"{basename.split('_LLIO')[0]}", fontsize=16)
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
    # cnn_plot_path = os.path.join(PLOT_DIR, f"{basename.split('_LLIO.mat')[0]}_window_based_cnn.png")
    plt.savefig(f"results/figs/window-based/window_based_cnn_exp_{expID}.png", dpi=300)
    # plt.savefig(f"results/figs/hybrid/hybrid_exp_{expID}.png", dpi=300)
    plt.pause(1)  # Show the figure for 1 second
    plt.close()   # Close the figure automatically

    # 3) Window-based Transformer data-driven INS grafiği
    plt.figure(figsize=(10, 6))
    plt.scatter(GCP[:,0], GCP[:,1], c='orange', marker='s', s=50, edgecolors='k', label='GCP')
    plt.plot(GCP[:,0], GCP[:,1], 'orange', linestyle='--', linewidth=1.5)
    plt.plot(prediction_trans_wcf[:,0], prediction_trans_wcf[:,1], 'b.-', linewidth=2, label='Window-based Transformer')
    # if expID not in [1, 51, 52, 53, 54]:
    if expID not in [51, 53, 54, 65, 66, 68, 75]:
        plt.plot(pyshoe_trajectory_wcf[:,0], pyshoe_trajectory_wcf[:,1], 'r.-', linewidth=1.5, label='PyShoe (LSTM)')
    plt.title(f"{basename.split('_LLIO')[0]}", fontsize=16)
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
    trans_plot_path = os.path.join(PLOT_DIR, f"{basename.split('_LLIO.mat')[0]}_window_based_transformer.png")
    # plt.savefig(trans_plot_path, dpi=300)
    plt.savefig(f"results/figs/window-based/window_based_transformer_exp_{expID}.png", dpi=300)
    # plt.savefig(f"results/figs/hybrid/hybrid_exp_{expID}.png", dpi=300)
    plt.pause(1)  # Show the figure for 1 second
    plt.close()   # Close the figure automatically

print("\n[TAMAMLANDI] Tüm .mat dosyaları işlendi ve grafikler kaydedildi.")
