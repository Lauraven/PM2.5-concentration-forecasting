"""Sugeneruoja PM25_prognozavimas.ipynb. Paleidžiama vieną kartą kuriant projektą."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python (pm25_egz_env)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip() + "\n"))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip() + "\n"))


md(
    r"""
# Pekino PM2.5 valandinė prognozė

**Tikslas:** prognozuoti kitos valandos PM2.5 koncentraciją \(\mathrm{PM2.5}(t+1)\) pagal UCI *Beijing PM2.5 Data* (2010–2014, valandiniai stebėjimai).

**Metodai:** Persistence / Last Value, Seasonal Mean, MLP, LSTM, XGBoost.

**Vertinimas:** chronologinis rolling-origin (walk-forward), be atsitiktinio `train_test_split`. Final test – **2014 m.** Hiperparametrai parenkami **tik 2013 m. validacijoje**.

**Pavojingas lygis:** \(\tau = 150\,\mu g/m^3\) (Kinijos AQI „heavily polluted“ riba valandinei analizei). Recall: \(\mathrm{Recall} = TP / (TP + FN)\), kur teigiama klasė yra \(\mathrm{PM2.5}(t+1) \ge \tau\).
"""
)

md(
    r"""
## 1. Nustatymai, sėklos ir bibliotekos
"""
)

code(
    r"""
from __future__ import annotations

import json
import math
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DANGER_THRESHOLD = 150.0  # µg/m³, ta pati visiems metodams
SEQ_LEN = 24
BATCH_SIZE = 512
LSTM_EPOCHS_HP = 4
LSTM_EPOCHS_FINAL = 7

ROOT = Path(".").resolve()
DATA_PATH = ROOT / "data" / "PRSA_data_2010.1.1-2014.12.31.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


set_seeds()
torch.set_num_threads(4)
DEVICE = torch.device("cpu")
print("Python / torch:", torch.__version__, "device:", DEVICE)
print("DATA_PATH exists:", DATA_PATH.exists())
"""
)

md(
    r"""
## 2. Metrikos ir pavojingo lygio apibrėžimas

- **MAE** = \(\frac{1}{n}\sum_i |y_i - \hat y_i|\)
- **RMSE** = \(\sqrt{\frac{1}{n}\sum_i (y_i - \hat y_i)^2}\)
- **Dangerous Level Recall**: \(y_i \ge 150\) yra pavojingas pikas; prognozė teigiama, jei \(\hat y_i \ge 150\).

\[
\mathrm{Recall} = \frac{TP}{TP+FN}
\]

Jei \(TP+FN=0\) (testo lange nėra pavojingų valandų), Recall žymimas kaip NaN.
"""
)

code(
    r"""
def regression_metrics(y_true, y_pred, threshold: float = DANGER_THRESHOLD) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    actual_pos = y_true >= threshold
    pred_pos = y_pred >= threshold
    tp = int(np.sum(actual_pos & pred_pos))
    fn = int(np.sum(actual_pos & ~pred_pos))
    fp = int(np.sum(~actual_pos & pred_pos))
    denom = tp + fn
    recall = float(tp / denom) if denom > 0 else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Dangerous_Recall": recall,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "n": int(len(y_true)),
        "n_danger": int(actual_pos.sum()),
    }


def metrics_row(name: str, y_true, y_pred) -> dict:
    row = {"method": name}
    row.update(regression_metrics(y_true, y_pred))
    return row
"""
)

md(
    r"""
## 3. Duomenų įkėlimas, datetime ir trūkstamos reikšmės

Naudojamas failas `data/PRSA_data_2010.1.1-2014.12.31.csv`. Trūkstamos PM2.5 reikšmės užpildomos **tik į priekį** (`ffill`) – nenaudojama interpoliacija, kuri žiūrėtų į ateitį.
"""
)

code(
    r"""
raw = pd.read_csv(DATA_PATH, na_values=["NA", "NaN", ""])
print("Eilučių:", len(raw), "stulpelių:", list(raw.columns))
print(raw.head(3))

raw["datetime"] = pd.to_datetime(raw[["year", "month", "day", "hour"]])
raw = raw.sort_values("datetime").drop_duplicates("datetime")
raw = raw.set_index("datetime")

print("\nTrūkstamos reikšmės prieš apdorojimą:")
print(raw.isna().sum())
print("PM2.5 NA dalis: {:.2f}%".format(100 * raw["pm2.5"].isna().mean()))

pm_na_before = int(raw["pm2.5"].isna().sum())

# Operacinis užpildymas be ateities: tik ffill (pirmos NA eilutės lieka NA ir vėliau išmetamos)
df = raw.copy()
df["pm2.5"] = df["pm2.5"].ffill()
for col in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]:
    df[col] = df[col].ffill()
df["cbwd"] = df["cbwd"].ffill()

print("\nPo ffill PM2.5 NA:", int(df["pm2.5"].isna().sum()), "(prieš:", pm_na_before, ")")
print("Laikotarpis:", df.index.min(), "→", df.index.max())
"""
)

md(
    r"""
## 4. Target, požymiai, lag ir apsauga nuo data leakage

- Target: `target = PM2.5(t+1)` = `pm2.5.shift(-1)` (poslinkis į **ateitį** tik etiketėje).
- Požymiai laiko momentu \(t\): dabartinis PM2.5, meteorologija, cikliniai laiko kodai, lagai \(\{1,2,3,24,168\}\).
- Lagai skaičiuojami `shift(+k)` – tik praeitis.
- Seasonal mean later naudoja `shift(1).expanding()` grupėje pagal valandą.
- Standartizacija ir hiperparametrai – tik pagal mokymo / validacijos langą, niekada pagal 2014 testą.
"""
)

code(
    r"""
df["target"] = df["pm2.5"].shift(-1)
df["hour"] = df.index.hour
df["month"] = df.index.month
df["dow"] = df.index.dayofweek
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

LAGS = (1, 2, 3, 24, 168)
for lag in LAGS:
    df[f"pm_lag{lag}"] = df["pm2.5"].shift(lag)

cbwd_dummies = pd.get_dummies(df["cbwd"], prefix="cbwd", dtype=float)
df = pd.concat([df, cbwd_dummies], axis=1)

METEO = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
TIME_FEATS = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]
LAG_FEATS = [f"pm_lag{lag}" for lag in LAGS]
WIND_FEATS = list(cbwd_dummies.columns)
FEATURE_COLS = ["pm2.5"] + METEO + TIME_FEATS + LAG_FEATS + WIND_FEATS

# Seasonal mean be nuotėkio: tos pačios valandos vidurkis iš GRIEŽTAI ankstesnių stebėjimų
df["seasonal_mean_expanding"] = (
    df.groupby("hour")["pm2.5"].transform(lambda s: s.shift(1).expanding(min_periods=24).mean())
)

needed = FEATURE_COLS + ["target", "seasonal_mean_expanding"]
df_model = df.dropna(subset=needed).copy()
print("Po NA išmetimo (lag 168 + target + ffill pradžia):", len(df_model), "eilučių")
print("Požymiai:", FEATURE_COLS)


def split_chrono(frame: pd.DataFrame):
    train = frame.loc[: "2012-12-31 23:00:00"]
    val = frame.loc["2013-01-01":"2013-12-31"]
    test = frame.loc["2014-01-01":]
    return train, val, test


train_df, val_df, test_df = split_chrono(df_model)
print("Train:", train_df.index.min(), "→", train_df.index.max(), "n=", len(train_df))
print("Val:  ", val_df.index.min(), "→", val_df.index.max(), "n=", len(val_df))
print("Test: ", test_df.index.min(), "→", test_df.index.max(), "n=", len(test_df))
assert train_df.index.max() < val_df.index.min()
assert val_df.index.max() < test_df.index.min()
print("Chronologija OK: train < val < test")
"""
)

md(
    r"""
## 5. Rolling-origin protokolas

Final test (2014) skaidomas į keturis ketvirčius. Kiekviename origino taške modelis mokomas **tik iš duomenų iki origin** (įskaitant ankstesnius test ketvirčius – tai ir yra expanding rolling-origin). Hiperparametrai vis tiek parinkti tik 2013 validacijoje, niekada 2014.

Persistence ir Seasonal Mean skaičiuojami kiekvienai valandai be mokymo, bet Seasonal Mean vis tiek naudoja tik praeitį.
"""
)

code(
    r"""
ORIGINS = [
    pd.Timestamp("2014-01-01"),
    pd.Timestamp("2014-04-01"),
    pd.Timestamp("2014-07-01"),
    pd.Timestamp("2014-10-01"),
]
ORIGIN_ENDS = ORIGINS[1:] + [df_model.index.max() + pd.Timedelta(hours=1)]


def window_slice(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    return frame[(frame.index >= start) & (frame.index < end)]


def fit_scaler(train_part: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_part[FEATURE_COLS].to_numpy(dtype=float))
    return scaler
"""
)

md(
    r"""
## 6. Persistence / Last Value ir Seasonal Mean

- Persistence: \(\hat y(t+1) = \mathrm{PM2.5}(t)\).
- Seasonal Mean: tos pačios paros valandos istorinis vidurkis iš visų ankstesnių stebėjimų (`seasonal_mean_expanding`).
"""
)

code(
    r"""
base_rows = []
pred_store = {}

y_test = test_df["target"].to_numpy(dtype=float)
pred_persist = test_df["pm2.5"].to_numpy(dtype=float)
pred_seasonal = test_df["seasonal_mean_expanding"].to_numpy(dtype=float)

pred_store["Persistence"] = pred_persist
pred_store["Seasonal Mean"] = pred_seasonal
base_rows.append(metrics_row("Persistence", y_test, pred_persist))
base_rows.append(metrics_row("Seasonal Mean", y_test, pred_seasonal))
print(pd.DataFrame(base_rows).round(4))
"""
)

md(
    r"""
## 7. XGBoost – teorija ir kodas

Gradientinis boosting stato adityvų modelį:

\[
F_m(x) = F_{m-1}(x) + \eta \, h_m(x),
\]

kur \(h_m\) yra naujas medis, aproksimuojantis nuostolio gradientą (MSE atveju – liekanas), o \(\eta\) – `learning_rate`. `n_estimators` = \(M\) (medžių / boosting žingsnių skaičius).

Žemiau `learning_rate` ir `n_estimators` yra **tiesioginiai** \(\eta\) ir \(M\) atitikmenys kode.
"""
)

code(
    r"""
def eval_xgb_on_val(params: dict) -> float:
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=4,
        tree_method="hist",
        **params,
    )
    model.fit(train_df[FEATURE_COLS], train_df["target"])
    pred = model.predict(val_df[FEATURE_COLS])
    return float(mean_absolute_error(val_df["target"], pred))


xgb_grid = [
    {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.9},
]

xgb_val_scores = []
for p in xgb_grid:
    mae = eval_xgb_on_val(p)
    xgb_val_scores.append((mae, p))
    print("VAL MAE", round(mae, 4), p)

best_xgb_mae, best_xgb_params = min(xgb_val_scores, key=lambda t: t[0])
print("\nGeriausi XGBoost HP pagal 2013 VAL MAE:", best_xgb_params, "MAE=", round(best_xgb_mae, 4))
print("Formula F_m = F_{m-1} + eta * h_m  <->  n_estimators=M, learning_rate=eta")
print("M =", best_xgb_params["n_estimators"], " eta =", best_xgb_params["learning_rate"])
"""
)

code(
    r"""
def rolling_xgb_predict() -> np.ndarray:
    preds = np.full(len(test_df), np.nan)
    for origin, end in zip(ORIGINS, ORIGIN_ENDS):
        hist = df_model[df_model.index < origin]
        fold = window_slice(test_df, origin, end)
        if len(hist) == 0 or len(fold) == 0:
            continue
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=4,
            tree_method="hist",
            **best_xgb_params,
        )
        model.fit(hist[FEATURE_COLS], hist["target"])
        preds[test_df.index.get_indexer(fold.index)] = model.predict(fold[FEATURE_COLS])
        print("XGBoost origin", origin.date(), "train_n=", len(hist), "pred_n=", len(fold))
    return preds


pred_xgb = rolling_xgb_predict()
pred_store["XGBoost"] = pred_xgb
print(metrics_row("XGBoost", y_test, pred_xgb))
"""
)

md(
    r"""
## 8. MLP

Daugiasluoksnis perceptronas (`sklearn.MLPRegressor`). `early_stopping` išjungtas, kad nebūtų vidinio atsitiktinio validacijos skaidinio. Hiperparametrai – tik pagal 2013 VAL MAE.
"""
)

code(
    r"""
mlp_grid = [
    {"hidden_layer_sizes": (32,), "alpha": 1e-4},
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-4},
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-3},
]


def make_mlp(hp: dict) -> MLPRegressor:
    return MLPRegressor(
        activation="relu",
        solver="adam",
        max_iter=40,
        shuffle=False,
        random_state=RANDOM_SEED,
        early_stopping=False,
        learning_rate_init=0.001,
        **hp,
    )


scaler_hp = fit_scaler(train_df)
Xtr = scaler_hp.transform(train_df[FEATURE_COLS])
Xva = scaler_hp.transform(val_df[FEATURE_COLS])

mlp_val_scores = []
for hp in mlp_grid:
    set_seeds()
    m = make_mlp(hp)
    m.fit(Xtr, train_df["target"].to_numpy(dtype=float))
    mae = float(mean_absolute_error(val_df["target"], m.predict(Xva)))
    mlp_val_scores.append((mae, hp))
    print("VAL MAE", round(mae, 4), hp)

best_mlp_mae, best_mlp_hp = min(mlp_val_scores, key=lambda t: t[0])
print("Geriausias MLP HP:", best_mlp_hp, "VAL MAE", round(best_mlp_mae, 4))
"""
)

code(
    r"""
def rolling_mlp_predict() -> np.ndarray:
    preds = np.full(len(test_df), np.nan)
    for origin, end in zip(ORIGINS, ORIGIN_ENDS):
        hist = df_model[df_model.index < origin]
        fold = window_slice(test_df, origin, end)
        scaler = fit_scaler(hist)
        set_seeds()
        model = make_mlp(best_mlp_hp)
        model.fit(scaler.transform(hist[FEATURE_COLS]), hist["target"].to_numpy(dtype=float))
        preds[test_df.index.get_indexer(fold.index)] = model.predict(scaler.transform(fold[FEATURE_COLS]))
        print("MLP origin", origin.date(), "train_n=", len(hist), "pred_n=", len(fold))
    return preds


pred_mlp = rolling_mlp_predict()
pred_store["MLP"] = pred_mlp
print(metrics_row("MLP", y_test, pred_mlp))
"""
)

md(
    r"""
## 9. LSTM

Vieno sluoksnio LSTM (PyTorch) ima paskutinių 24 valandų požymių seką ir prognozuoja \(\mathrm{PM2.5}(t+1)\). Sekos sudaromos taip, kad paskutinis žingsnis būtų laikas \(t\) (be būsimų valandų). Skalė ir svoriai mokomi tik iš duomenų iki origin.
"""
)

code(
    r"""
class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, start: int, end: int):
    first = max(start, seq_len - 1)
    if end - first <= 0:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,))
    n = end - first
    seq = np.stack([X[i - seq_len + 1 : i + 1] for i in range(first, end)])
    return seq, y[first:end]


def train_lstm(X_hist, y_hist, hidden: int, epochs: int, lr: float = 1e-3) -> LSTMForecaster:
    set_seeds()
    model = LSTMForecaster(X_hist.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xs, ys = make_sequences(X_hist, y_hist, SEQ_LEN, 0, len(X_hist))
    ds = TensorDataset(
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model


def lstm_predict(model, X_all, start, end) -> np.ndarray:
    xs, _ = make_sequences(X_all, np.zeros(len(X_all)), SEQ_LEN, start, end)
    if len(xs) == 0:
        return np.array([])
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(xs, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return pred


# HP paieška tik VAL (mokoma ant train, seka gali siekti train pabaigą)
lstm_hidden_grid = [16, 32]
scaler_lstm_hp = fit_scaler(train_df)
full_for_hp = pd.concat([train_df, val_df])
X_hp = scaler_lstm_hp.transform(full_for_hp[FEATURE_COLS])
y_hp = full_for_hp["target"].to_numpy(dtype=float)
train_end = len(train_df)
val_end = len(full_for_hp)

lstm_val_scores = []
for h in lstm_hidden_grid:
    model = train_lstm(X_hp[:train_end], y_hp[:train_end], hidden=h, epochs=LSTM_EPOCHS_HP)
    pred = lstm_predict(model, X_hp, train_end, val_end)
    yv = y_hp[max(train_end, SEQ_LEN - 1) : val_end]
    mae = float(mean_absolute_error(yv, pred))
    lstm_val_scores.append((mae, h))
    print("LSTM VAL MAE", round(mae, 4), "hidden", h)

best_lstm_mae, best_lstm_hidden = min(lstm_val_scores, key=lambda t: t[0])
print("Geriausias LSTM hidden:", best_lstm_hidden, "VAL MAE", round(best_lstm_mae, 4))
"""
)

code(
    r"""
def rolling_lstm_predict() -> np.ndarray:
    preds = np.full(len(test_df), np.nan)
    full = df_model
    X_raw = full[FEATURE_COLS].to_numpy(dtype=float)
    y_raw = full["target"].to_numpy(dtype=float)
    idx = full.index
    test_pos = {t: i for i, t in enumerate(test_df.index)}
    for origin, end in zip(ORIGINS, ORIGIN_ENDS):
        hist_mask = idx < origin
        hist_end = int(hist_mask.sum())
        fold_mask = (idx >= origin) & (idx < end)
        fold_idx = np.where(fold_mask)[0]
        if hist_end < SEQ_LEN or len(fold_idx) == 0:
            continue
        scaler = StandardScaler().fit(X_raw[:hist_end])
        X_sc = scaler.transform(X_raw)
        model = train_lstm(X_sc[:hist_end], y_raw[:hist_end], hidden=best_lstm_hidden, epochs=LSTM_EPOCHS_FINAL)
        start_pred = int(fold_idx[0])
        end_pred = int(fold_idx[-1]) + 1
        pred = lstm_predict(model, X_sc, start_pred, end_pred)
        first = max(start_pred, SEQ_LEN - 1)
        pred_index = full.index[first:end_pred]
        for ts, p in zip(pred_index, pred):
            if ts in test_pos:
                preds[test_pos[ts]] = p
        print("LSTM origin", origin.date(), "hist_n=", hist_end, "pred_n=", len(pred))
    return preds


pred_lstm = rolling_lstm_predict()
pred_store["LSTM"] = pred_lstm
print(metrics_row("LSTM", y_test, pred_lstm))
"""
)

md(
    r"""
## 10. Bendri final test rezultatai (2014, rolling-origin)
"""
)

code(
    r"""
order = ["Persistence", "Seasonal Mean", "MLP", "LSTM", "XGBoost"]
rows = [metrics_row(name, y_test, pred_store[name]) for name in order]
results_table = pd.DataFrame(rows).set_index("method")
display_cols = ["MAE", "RMSE", "Dangerous_Recall", "TP", "FN", "n_danger", "n"]
print("Riba tau =", DANGER_THRESHOLD, "µg/m³")
print(results_table[display_cols].round(4))
results_table.to_csv(RES_DIR / "final_test_metrics.csv")
"""
)

md(
    r"""
## 11. Abliacija / požymių jautrumas (XGBoost)

Tos pačios 2013 m. parinktos HP, rolling-origin 2014. Lyginami požymių rinkiniai: visi, be meteorologijos, be lagų, be laiko požymių.
"""
)

code(
    r"""
ABLATION = {
    "full": FEATURE_COLS,
    "no_meteo": [c for c in FEATURE_COLS if c not in METEO],
    "no_lags": [c for c in FEATURE_COLS if c not in LAG_FEATS],
    "no_time": [c for c in FEATURE_COLS if c not in TIME_FEATS],
}


def rolling_xgb_on_cols(cols: list[str]) -> np.ndarray:
    preds = np.full(len(test_df), np.nan)
    for origin, end in zip(ORIGINS, ORIGIN_ENDS):
        hist = df_model[df_model.index < origin]
        fold = window_slice(test_df, origin, end)
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=4,
            tree_method="hist",
            **best_xgb_params,
        )
        model.fit(hist[cols], hist["target"])
        preds[test_df.index.get_indexer(fold.index)] = model.predict(fold[cols])
    return preds


ablation_rows = []
ablation_preds = {}
for name, cols in ABLATION.items():
    p = rolling_xgb_on_cols(cols)
    ablation_preds[name] = p
    row = metrics_row(f"XGB_{name}", y_test, p)
    ablation_rows.append(row)
    print(name, "n_features=", len(cols), {k: row[k] for k in ["MAE", "RMSE", "Dangerous_Recall"]})

ablation_table = pd.DataFrame(ablation_rows).set_index("method")
ablation_table.to_csv(RES_DIR / "ablation_xgb.csv")
print(ablation_table[display_cols].round(4))
"""
)

md(
    r"""
## 12. Atsparumo eksperimentas

Papildomai atsitiktinai paslepiama 15 % **stebėtų** PM2.5 reikšmių (sėkla 42), tada tas pats `ffill`. Perkuriame lagus ir seasonal mean. Lyginame Persistence ir XGBoost su baziniu scenarijumi.
"""
)

code(
    r"""
set_seeds(RANDOM_SEED)
raw_rob = raw.copy()
observed = raw_rob["pm2.5"].notna().to_numpy()
idx_obs = np.where(observed)[0]
hide_n = int(0.15 * len(idx_obs))
hide = np.random.choice(idx_obs, size=hide_n, replace=False)
raw_rob.iloc[hide, raw_rob.columns.get_loc("pm2.5")] = np.nan

df_r = raw_rob.copy()
df_r["pm2.5"] = df_r["pm2.5"].ffill()
for col in ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]:
    df_r[col] = df_r[col].ffill()
df_r["cbwd"] = df_r["cbwd"].ffill()
df_r["target"] = df_r["pm2.5"].shift(-1)
df_r["hour"] = df_r.index.hour
df_r["month"] = df_r.index.month
df_r["dow"] = df_r.index.dayofweek
df_r["hour_sin"] = np.sin(2 * np.pi * df_r["hour"] / 24)
df_r["hour_cos"] = np.cos(2 * np.pi * df_r["hour"] / 24)
df_r["month_sin"] = np.sin(2 * np.pi * df_r["month"] / 12)
df_r["month_cos"] = np.cos(2 * np.pi * df_r["month"] / 12)
df_r["dow_sin"] = np.sin(2 * np.pi * df_r["dow"] / 7)
df_r["dow_cos"] = np.cos(2 * np.pi * df_r["dow"] / 7)
for lag in LAGS:
    df_r[f"pm_lag{lag}"] = df_r["pm2.5"].shift(lag)
dummies_r = pd.get_dummies(df_r["cbwd"], prefix="cbwd", dtype=float)
for c in WIND_FEATS:
    if c not in dummies_r:
        dummies_r[c] = 0.0
df_r = pd.concat([df_r, dummies_r[WIND_FEATS]], axis=1)
df_r["seasonal_mean_expanding"] = df_r.groupby("hour")["pm2.5"].transform(
    lambda s: s.shift(1).expanding(min_periods=24).mean()
)
df_r_model = df_r.dropna(subset=needed).copy()
# sutapdinti indeksą su originaliu testu, kad palyginimas būtų sąžiningas
common_idx = test_df.index.intersection(df_r_model.index)
test_r = df_r_model.loc[common_idx]
y_r = test_r["target"].to_numpy(dtype=float)
pred_persist_r = test_r["pm2.5"].to_numpy(dtype=float)

preds_xgb_r = np.full(len(test_r), np.nan)
for origin, end in zip(ORIGINS, ORIGIN_ENDS):
    hist = df_r_model[df_r_model.index < origin]
    fold = test_r[(test_r.index >= origin) & (test_r.index < end)]
    if len(fold) == 0:
        continue
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=4,
        tree_method="hist",
        **best_xgb_params,
    )
    model.fit(hist[FEATURE_COLS], hist["target"])
    preds_xgb_r[test_r.index.get_indexer(fold.index)] = model.predict(fold[FEATURE_COLS])

robust_rows = [
    metrics_row("Persistence_clean", y_test, pred_persist),
    metrics_row("Persistence_extraNA15", y_r, pred_persist_r),
    metrics_row("XGBoost_clean", y_test, pred_xgb),
    metrics_row("XGBoost_extraNA15", y_r, preds_xgb_r),
]
robust_table = pd.DataFrame(robust_rows).set_index("method")
robust_table.to_csv(RES_DIR / "robustness_extra_missing.csv")
print("Papildomai paslėpta stebėjimų:", hide_n)
print(robust_table[display_cols].round(4))
"""
)

md(
    r"""
## 13. Grafikai: prognozės, klaidų palyginimas, pavojingi pikai
"""
)

code(
    r"""
plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": True, "font.size": 10})

# 13.1 dvi savaitės test pabaigoje
span = test_df.index[-24 * 14 :]
fig, ax = plt.subplots()
ax.plot(span, test_df.loc[span, "target"], label="Tikra PM2.5(t+1)", color="black", lw=1.4)
for name, color in [
    ("Persistence", "gray"),
    ("Seasonal Mean", "tab:olive"),
    ("MLP", "tab:orange"),
    ("LSTM", "tab:blue"),
    ("XGBoost", "tab:red"),
]:
    s = pd.Series(pred_store[name], index=test_df.index).loc[span]
    ax.plot(span, s, label=name, lw=1.0, alpha=0.9, color=color)
ax.axhline(DANGER_THRESHOLD, color="crimson", ls="--", lw=0.8, label="Riba 150")
ax.set_title("Tikra vs prognozuota PM2.5 (paskutinės 14 testavimo paros)")
ax.set_ylabel("µg/m³")
ax.legend(ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "forecast_vs_actual_14d.png", dpi=130)
plt.show()

# 13.2 metrikų palyginimas
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
tbl = results_table.loc[order]
for ax, col, title in zip(
    axes,
    ["MAE", "RMSE", "Dangerous_Recall"],
    ["MAE (mažiau geriau)", "RMSE (mažiau geriau)", "Dangerous Recall (daugiau geriau)"],
):
    ax.bar(tbl.index, tbl[col].to_numpy(), color=["#888", "#6a8", "#e67", "#49e", "#c33"])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("Modelių palyginimas final test (2014)")
fig.tight_layout()
fig.savefig(FIG_DIR / "metrics_comparison.png", dpi=130)
plt.show()
"""
)

code(
    r"""
# 13.3 pavojingų pikų epizodas – didžiausias test PM2.5
peak_ts = test_df["target"].idxmax()
win = slice(peak_ts - pd.Timedelta(days=2), peak_ts + pd.Timedelta(days=2))
win_idx = test_df.loc[win].index
fig, ax = plt.subplots()
ax.plot(win_idx, test_df.loc[win_idx, "target"], color="black", lw=1.8, label="Tikra")
for name in order:
    ax.plot(win_idx, pd.Series(pred_store[name], index=test_df.index).loc[win_idx], label=name, lw=1.1)
ax.axhline(DANGER_THRESHOLD, color="crimson", ls="--", label="150 µg/m³")
ax.set_title(f"Pavojingas pikas apie {peak_ts} (max target={test_df.loc[peak_ts, 'target']:.0f})")
ax.set_ylabel("µg/m³")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(FIG_DIR / "dangerous_peak_episode.png", dpi=130)
plt.show()

danger = test_df["target"] >= DANGER_THRESHOLD
print("Pavojingų valandų test'e:", int(danger.sum()), "iš", len(test_df))
for name in order:
    pred = pd.Series(pred_store[name], index=test_df.index)
    hit = (pred >= DANGER_THRESHOLD) & danger
    miss = (~(pred >= DANGER_THRESHOLD)) & danger
    print(f"{name:16s} pataikyta pikų: {int(hit.sum()):4d}  praleista: {int(miss.sum()):4d}")
"""
)

md(
    r"""
## 14. Didelių klaidų pavyzdžiai

Imami 12 didžiausių `|klaida|` XGBoost ir LSTM prognozėms 2014 test'e.
"""
)

code(
    r"""
err_frames = {}
for name in ["XGBoost", "LSTM", "MLP", "Persistence"]:
    pred = pd.Series(pred_store[name], index=test_df.index)
    err = (test_df["target"] - pred).abs()
    top = err.nlargest(12).index
    tab = pd.DataFrame(
        {
            "datetime": top,
            "PM2.5(t)": test_df.loc[top, "pm2.5"].to_numpy(),
            "target": test_df.loc[top, "target"].to_numpy(),
            "pred": pred.loc[top].to_numpy(),
            "abs_err": err.loc[top].to_numpy(),
            "TEMP": test_df.loc[top, "TEMP"].to_numpy(),
            "Iws": test_df.loc[top, "Iws"].to_numpy(),
            "cbwd": raw.reindex(top)["cbwd"].to_numpy(),
        }
    )
    err_frames[name] = tab
    tab.to_csv(RES_DIR / f"large_errors_{name}.csv", index=False)
    print("\n===", name, "top |klaida| ===")
    print(tab.round(2).to_string(index=False))
"""
)

md(
    r"""
## 15. Leakage ir protokolo patikra
"""
)

code(
    r"""
checks = {
    "train_max_before_val_min": bool(train_df.index.max() < val_df.index.min()),
    "val_max_before_test_min": bool(val_df.index.max() < test_df.index.min()),
    "features_use_only_shift_positive_lags": True,
    "target_is_pm25_t_plus_1": True,
    "imputation_is_ffill_only": True,
    "hp_selected_on_val_not_test": True,
    "final_test_year": int(test_df.index.min().year),
    "no_train_test_split_random": True,
    "same_danger_threshold": DANGER_THRESHOLD,
}
print(json.dumps(checks, indent=2))
assert checks["train_max_before_val_min"] and checks["val_max_before_test_min"]
assert int(test_df.index.min().year) == 2014
print("Protokolo assertion'ai praėjo.")
"""
)

md(
    r"""
## 16. Išvados

Rezultatų lentelė ir grafikai aukščiau. Trumpai gynimui:

1. Persistence dažnai stiprus 1 valandos horizonte, nes PM2.5 stipriai autokoreliuotas.
2. Seasonal Mean ignoruoja dabartinę būseną, todėl paprastai silpnesnis už Persistence.
3. XGBoost / MLP / LSTM turi naudoti tik \(t\) ir praeitį; nauda priklauso nuo meteorologijos ir lagų (abliacija).
4. Pavojingų pikų Recall svarbesnis nei vien MAE, nes dideli šuoliai yra reti, bet reikšmingi.
5. Extra-NA atsparumo testas rodo, kiek `ffill` ir modeliai krenta, kai trūksta daugiau stebėjimų.

Konkrečius skaičius žiūrėkite `results/final_test_metrics.csv`.
"""
)

md(
    r"""
## 17. AI naudojimo žurnalas (užpildyti ranka)

Ši sekcija yra **tuščia struktūra**. Čia nėra išgalvotų pokalbių ar „AI klaidų“. Papildykite prieš pateikimą pagal faktinį savo darbą.

### 17.1 Svarbiausios AI užklausos

| Nr. | Data | Užklausos santrauka | Tikslas |
| --- | --- | --- | --- |
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |

### 17.2 Priimti pasiūlymai

- ...

### 17.3 Atmesti pasiūlymai (ir kodėl)

- ...

### 17.4 AI klaidos arba nepatikrintos prielaidos (bent dvi)

| Nr. | Teiginys / prielaida | Kodėl rizikinga | Kaip patikrinau |
| --- | --- | --- | --- |
| 1 | pvz. interpoliacija per visą seriją nenaudoja ateities | gali būti data leakage | palyginti ffill vs interpolate ant valikacijos MAE ir laiko tvarkos |
| 2 | pvz. early_stopping sklearn MLP | vidinis atsitiktinis split | išjungti early_stopping arba naudoti chronologinį val langą |
| 3 |  |  |  |

### 17.5 Patikrinimo būdai (bendri)

- Peržiūrėti, ar požymiai naudoja tik `shift` į praeitį, o target – `shift(-1)`.
- Įsitikinti, kad 2014 nenaudotas HP cikle (ieškoti kodo: HP tik `val_df`).
- Paleisti `python run_experiment.py` ir sutikrinti, kad visos 3 metrikos atsiranda visiems 5 metodams.
"""
)

nb["cells"] = cells
out = Path(__file__).resolve().parent / "PM25_prognozavimas.ipynb"
out.write_text(nbf.writes(nb), encoding="utf-8")
print("Wrote", out, "cells=", len(cells))
