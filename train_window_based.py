import os
import random
import joblib
import numpy as np
from glob import glob
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# ======================================
# --- AYARLAR & SABİTLER (HYPERPARAMS) ---
# ======================================
SEED       = 120
DATA_DIR   = "data/training"  # .mat dosyalarının olduğu dizin
OUTPUT_DIR = "results/window-based"  # Model ve scaler çıktılarının kaydedileceği dizin
MAX_LEN    = 256
PATIENCE   = 15
MAX_EPOCHS = 1000
AUG_FACTOR = 8
NOISE_STD  = 0.02

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark    = False


# ======================================================
# --- YARDIMCI FONKSİYONLAR & VERİ HAZIRLAMA ADIMLARI ---
# ======================================================

def pad_truncate(seg: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """
    Bir IMU segmentini max_len uzunluğuna getirir:
    - Eğer segment uzunluğu >= max_len ise, baştan keser.
    - Eğer segment uzunluğu < max_len ise, son satırı tekrarlayarak doldurur.
    """
    L = seg.shape[0]
    if L >= max_len:
        return seg[:max_len]
    pad = np.tile(seg[-1:], (max_len - L, 1))
    return np.vstack([seg, pad])  # (max_len, 3)


def collect_and_augment(mat_dir: str):
    """
    Verilen klasördeki tüm .mat dosyalarını tarar, her bir stride segmentini çıkarır.
    Sadece ivme (acc) verisi kullanılır (gyroskop yok). Her segment için dx, dy hesaplanır.
    Hızın 90. persentili üzerindeki segmentlere gürültü eklenerek augmentasyon uygulanır.
    Geri dönüş:
      raw: [(seg_a (N×3), src_flag, dx, dy, speed), ...]
      thr: 90. persentile göre belirlenen hız eşiği (float)
    """
    raw = []
    for fp in sorted(glob(os.path.join(mat_dir, "*.mat"))):
        d        = loadmat(fp)
        idx      = d["strideIndex"].flatten()     # (M+1,) stride indeksleri
        acc      = d["acc_n"]                     # (TotalN × 3) ivme
        GCP      = d["GCP"]                       # (M × 2) ground-truth koordinatlar
        src_flag = 1 if "SensorConnectData" in os.path.basename(fp) else 0

        # Her iki ardışık strideIndex çifti (s,e) için bir segment
        for i in range(len(idx) - 1):
            s, e = idx[i], idx[i + 1]
            if s < 0 or e > acc.shape[0] or e <= s:
                continue

            seg_a = acc[s:e, :]   # (segment_length × 3)
            if seg_a.shape[0] < 10:
                continue

            # Ground-truth dx, dy
            dx = float(GCP[i + 1, 0] - GCP[i, 0])
            dy = float(GCP[i + 1, 1] - GCP[i, 1])

            # Süre (saniye), sensör frekansı 200 Hz
            duration = (e - s) / 200.0
            speed = dx / duration if duration > 0 else 0.0

            raw.append((seg_a, src_flag, dx, dy, speed))

    # Hız eşiğini bul (90. persentil)
    speeds = np.array([r[4] for r in raw])
    thr    = np.quantile(speeds, 0.9)

    # 90. persentilin üzerindeki segmentleri augment et (gürültü ekleyerek)
    fast   = [r for r in raw if r[4] >= thr]
    for seg_a, src, dx, dy, speed in fast:
        for _ in range(AUG_FACTOR):
            na = seg_a + np.random.normal(0, NOISE_STD * np.abs(seg_a), seg_a.shape)
            raw.append((na, src, dx, dy, speed))

    return raw, thr


class IMUDataset(Dataset):
    """
    Yalnızca ivme (acc) verisi + src_flag kullanarak (dx, dy) etiketine sahip PyTorch Dataset.
    Eğer fit_scalers=True verilirse, x_scaler ve y_scaler scaler'ları bu veri üzerinde fit edilir.
    """
    def __init__(self, data, x_scaler: StandardScaler, y_scaler: StandardScaler, fit_scalers: bool = False):
        X_list, Y_list = [], []

        # data: [(seg_a (n×3), src_flag, dx, dy, speed), ...]
        for seg_a, src, dx, dy, _ in data:
            # 1) Pad/Truncate ivme segmentini (n×3) => (MAX_LEN×3)
            a = pad_truncate(seg_a)  # (MAX_LEN, 3)

            # 2) src_flag sütunu (MAX_LEN×1)
            f = np.full((MAX_LEN, 1), src, dtype=np.float32)

            # 3) Özellikleri birleştir: [acc_x, acc_y, acc_z, src_flag]
            X_list.append(np.concatenate([a.astype(np.float32), f], axis=1))  # (MAX_LEN, 4)
            Y_list.append([dx, dy])  # (2,)

        X = np.stack(X_list)                 # (num_samples, MAX_LEN, 4)
        Y = np.array(Y_list, dtype=np.float32)  # (num_samples, 2)

        if fit_scalers:
            # Scaler'ları eğit (flatten ederek)
            x_scaler.fit(X.reshape(-1, 4))
            y_scaler.fit(Y)

        # Ölçekle
        Xs = x_scaler.transform(X.reshape(-1, 4)).reshape(X.shape).astype(np.float32)
        Ys = y_scaler.transform(Y).astype(np.float32)

        self.X, self.Y = Xs, Ys

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


# ====================
# --- MODEL TANIMLARI ---
# ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # x: (batch_size, MAX_LEN, d_model)
        return x + self.pe.to(x.device)


class IMUTransformer(nn.Module):
    def __init__(self, in_dim: int = 4, model_dim: int = 128, num_heads: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, model_dim)
        self.pos_enc    = PositionalEncoding(model_dim)
        enc_layer      = nn.TransformerEncoderLayer(
            d_model        = model_dim,
            nhead          = num_heads,
            dim_feedforward= model_dim*2,
            dropout        = dropout,
            batch_first    = True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(model_dim,   model_dim//2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim//2, 2)
        )

    def forward(self, x):
        # x: (batch_size, MAX_LEN, in_dim=4)
        x = self.input_proj(x)       # (batch_size, MAX_LEN, model_dim)
        x = self.pos_enc(x)          # (batch_size, MAX_LEN, model_dim)
        x = self.transformer(x)      # (batch_size, MAX_LEN, model_dim)
        return self.fc(x.mean(dim=1))  # (batch_size, 2)


class IMULSTM(nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = in_dim,
            hidden_size= hidden_dim,
            num_layers = num_layers,
            batch_first= True,
            dropout    = dropout,
            bidirectional = False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, x):
        # x: (batch_size, MAX_LEN, in_dim=4)
        out, _ = self.lstm(x)       # out: (batch_size, MAX_LEN, hidden_dim)
        return self.fc(out[:, -1, :])  # son zaman adımı (batch_size, hidden_dim) -> (batch_size, 2)


class IMUCNN(nn.Module):
    def __init__(self, in_ch: int = 4, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(num_layers):
            layers += [
                nn.Conv1d(ch, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ]
            ch = hidden_dim

        self.conv = nn.Sequential(*layers)
        fc_input_dim = hidden_dim * (MAX_LEN // (2**num_layers))
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, x):
        # x: (batch_size, MAX_LEN, in_ch=4) -> (batch_size, 4, MAX_LEN)
        x = x.permute(0, 2, 1)
        x = self.conv(x)      # (batch_size, hidden_dim, MAX_LEN/(2**num_layers))
        x = x.flatten(1)      # (batch_size, fc_input_dim)
        return self.fc(x)     # (batch_size, 2)


# =======================================
# --- VERİ HAZIRLAMA & SCALER OLUŞTURMA ---
# =======================================

# 1) Ham verileri topla ve augment et
raw, speed_thr = collect_and_augment(DATA_DIR)
joblib.dump(speed_thr, os.path.join(OUTPUT_DIR, "speed_thr.pkl"))

# 2) Train/Validation/Test split
train_r, tmp  = train_test_split(raw, test_size=0.3, random_state=SEED)
val_r, test_r = train_test_split(tmp,  test_size=0.5, random_state=SEED)

# 3) Scaler’ları yalnızca train seti üzerinde fit et
x_scaler = StandardScaler()
y_scaler = StandardScaler()
_ = IMUDataset(train_r, x_scaler, y_scaler, fit_scalers=True)
joblib.dump(x_scaler, os.path.join(OUTPUT_DIR, "x_scaler.pkl"))
joblib.dump(y_scaler, os.path.join(OUTPUT_DIR, "y_scaler.pkl"))

# 4) Dataset lambda fonksiyonları (veriyi DataLoader’a sarmak için)
base_train_ds = lambda bs: IMUDataset(train_r, x_scaler, y_scaler, fit_scalers=False)
base_val_ds   = lambda bs: IMUDataset(val_r,   x_scaler, y_scaler, fit_scalers=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# --- HYPERPARAMETER GRID & MAP ---
# ============================
model_map = {
    'transformer': IMUTransformer,
    'lstm':        IMULSTM,
    'cnn':         IMUCNN
}

param_grid = {
    'model': ['transformer', 'lstm', 'cnn'],
    'lr':    [1e-3, 5e-4],
    'batch_size': [32, 64]
}

# Her model tipi için ayrı ayrı "en iyi RMSE" ve "en iyi konfigürasyon" bilgisi
best_rmse_per_model = {
    'transformer': np.inf,
    'lstm':        np.inf,
    'cnn':         np.inf
}
best_cfg_per_model = {
    'transformer': None,
    'lstm':        None,
    'cnn':         None
}

# =======================
# --- GRID SEARCH BAŞLA ---
# =======================
for cfg in ParameterGrid(param_grid):
    name = cfg['model']       # 'transformer', 'lstm' veya 'cnn'
    lr   = cfg['lr']
    bs   = cfg['batch_size']
    print(f"\n>> Testing cfg: model={name}, lr={lr}, batch_size={bs}")

    # 1) DataLoader’ları hazırla
    train_ds = base_train_ds(bs)
    val_ds   = base_val_ds(bs)
    tr_lo = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True)
    va_lo = DataLoader(val_ds,   batch_size=bs*2, shuffle=False)

    # 2) Modeli instantiate et
    model = model_map[name]().to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, verbose=False)
    loss_fn = nn.MSELoss()

    # 3) Erken durdurma için takip değişkenleri
    cfg_best_rmse = np.inf
    wait = 0

    for ep in range(1, MAX_EPOCHS + 1):
        # --- TRAIN AŞAMASI ---
        model.train()
        for Xb, Yb in tr_lo:
            Xb, Yb = Xb.to(device), Yb.to(device)
            opt.zero_grad()
            loss_fn(model(Xb), Yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # --- VALIDATION AŞAMASI ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, Yb in va_lo:
                Xb, Yb = Xb.to(device), Yb.to(device)
                val_losses.append(loss_fn(model(Xb), Yb).item())
        mse = np.mean(val_losses)
        rmse = np.sqrt(mse)
        sched.step(mse)

        # Early stopping kontrolü
        if rmse < cfg_best_rmse - 1e-6:
            cfg_best_rmse, wait = rmse, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    print(f"Result cfg: model={name}, lr={lr}, bs={bs} -> val_RMSE={cfg_best_rmse:.6f}")

    # -------------------------------
    # Bu model tipine ait en iyi RMSE ile karşılaştır
    # -------------------------------
    if cfg_best_rmse < best_rmse_per_model[name]:
        best_rmse_per_model[name] = cfg_best_rmse
        best_cfg_per_model[name] = {'lr': lr, 'batch_size': bs}

        # TorchScript olarak kaydet
        example = torch.randn(1, MAX_LEN, 4, device=device)  # in_dim=4
        ts_mod  = torch.jit.trace(model, example, check_trace=False)
        model_filename = f"best_model_{name}_lr{lr}_bs{bs}.pt"
        ts_mod.save(os.path.join(OUTPUT_DIR, model_filename))
        print(f"  --> [{name}] için yeni en iyi model kaydedildi: {model_filename}")

# =============================
# --- GRID SEARCH SONU & ÖZET ---
# =============================
print("\n=== Grid Search Complete ===")
for m in ['transformer', 'lstm', 'cnn']:
    if best_cfg_per_model[m] is not None:
        lr_best   = best_cfg_per_model[m]['lr']
        bs_best   = best_cfg_per_model[m]['batch_size']
        rmse_best = best_rmse_per_model[m]
        print(f">> {m.upper():11s} | Best RMSE = {rmse_best:.6f} | LR = {lr_best}, BS = {bs_best}")
    else:
        print(f">> {m.upper():11s} | Hiç model kaydedilemedi.")
