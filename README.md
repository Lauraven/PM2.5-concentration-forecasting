# Pekino PM2.5 valandinė prognozė

Projektas prognozuoja **kitos valandos** PM2.5 koncentraciją (`PM2.5(t+1)`) pagal UCI *Beijing PM2.5 Data* rinkinį. Viename Jupyter notebook faile: duomenų paruošimas, baseline metodai (Persistence, Seasonal Mean), MLP, LSTM, XGBoost, chronologinis rolling-origin vertinimas, abliacija, atsparumo eksperimentas, grafikai, išvados ir AI naudojimo žurnalas.

## Python versija

Paruošta ir patikrinta su **Python 3.13.2** (64 bitų, Windows). Turėtų veikti ir su Python 3.11–3.13.

## Virtualios aplinkos sukūrimas ir aktyvavimas

**Windows PowerShell:**
```powershell
python -m venv pm25_egz_env
.\pm25_egz_env\Scripts\Activate.ps1
```

**Windows cmd:**
```bat
python -m venv pm25_egz_env
pm25_egz_env\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv pm25_egz_env
source pm25_egz_env/bin/activate
```

**Windows ir ilgi keliai.** Jei `pip` skundžiasi *Windows Long Path* (dažniausiai diegiant `torch`), sukurkite venv trumpame kelyje ir projekte palikite junction (pakeiskite kelią į savo):
```powershell
python -m venv C:\venvs\pm25_egz_env
cmd /c mklink /J pm25_egz_env C:\venvs\pm25_egz_env
.\pm25_egz_env\Scripts\Activate.ps1
```

## Priklausomybių įdiegimas

```powershell
python -m pip install -r requirements.txt
```

Jei `torch` diegimas iš `requirements.txt` nepavyksta, įdiekite CPU versiją atskirai:
```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

`requirements.txt` sąmoningai neįtraukia `jupyter`/JupyterLab meta-paketo (Windows su ilgu projekto keliu diegimas dažnai nutrūksta) — notebook paleidimui iš IDE užtenka `ipykernel` ir `nbconvert`.

## Duomenų atsisiuntimas

Duomenys turi būti: `data/PRSA_data_2010.1.1-2014.12.31.csv`

Jei failo nėra, atsisiųskite UCI rinkinį ir išarchyvuokite CSV į `data/`:
- https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data
- tiesioginis zip: https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip

## Notebook paleidimas

Aktyvavus `pm25_egz_env`, atidarykite `PM25_prognozavimas.ipynb` (VS Code / JupyterLab) ir paleiskite visas ląsteles (**Run All**). Branduolį rinkitės iš `pm25_egz_env`.

Kernelio registravimas (jei IDE aplinkos nematytų):
```powershell
python -m ipykernel install --user --name pm25_egz_env --display-name "Python (pm25_egz_env)"
```

## Viena komanda pagrindiniam eksperimentui pakartoti

Iš projekto šaknies, su aktyviu `pm25_egz_env`:
```powershell
python run_experiment.py
```
Skriptas įvykdo visą `PM25_prognozavimas.ipynb` (timeout 2 val.). Rezultatai įrašomi į `results/` ir `figures/`, notebook išsaugomas su išvestimi.

## Projekto struktūra

```
Egzaminas/
├── PM25_prognozavimas.ipynb   # visa analizė (vienintelis notebook)
├── run_experiment.py          # vienos komandos paleidimas
├── requirements.txt
├── README.md
├── pm25_egz_env/              # virtuali aplinka (sukuriama vietoje)
├── data/
│   └── PRSA_data_2010.1.1-2014.12.31.csv
├── figures/                   # grafikai (sukuria notebook)
└── results/                   # rezultatų lentelės CSV (sukuria notebook)
```

## Metodai, vertinimo protokolas ir metrikos

**Metodai:** Persistence (last value), Seasonal Mean (tos pačios valandos istorinis vidurkis), MLP, LSTM, PSO-LSTM ir XGBoost.

**Skaidymas — chronologinis, be atsitiktinio `train_test_split`:**

- Train: 2010–2012 · Val: 2013 (tik hiperparametrams rinktis) · Test: 2014 (galutinis, hiperparametrams niekada nenaudojamas).
- Test viduje — rolling-origin (walk-forward) vertinimas: 4 origin taškai (kas ketvirtį 2014 m.), kiekviename modelis pertreniruojamas tik iš iki tol turimos istorijos (expanding window).

**Metrikos** (visiems metodams vienodos): MAE, RMSE, Dangerous Level Recall = `TP / (TP + FN)`, riba **τ = 150 µg/m³**.

**Fiksuotos atsitiktinės sėklos:** `RANDOM_SEED = 42` naudojamas NumPy, PyTorch, scikit-learn ir XGBoost, kad rezultatai būtų atkuriami.

**Papildomi eksperimentai:**

- Abliacija (XGBoost, požymių grupės: pilnas / be meteo / be lag'ų / be laiko požymių) — `results/ablation_xgb.csv`.
- Atsparumo testas (papildomai pašalinta 15 % PM2.5 įvesties reikšmių) — `results/robustness_extra_missing.csv`.

## Rezultatai (final test, 2014 m.)

| **Metodas** | **MAE** | **RMSE** | **Dangerous Recall** | **TP** | **FN** |
|---|---:|---:|---:|---:|---:|
| Persistence | 11.9603 | 22.1377 | 0.9011 | 1586 | 174 |
| Seasonal Mean | 69.0668 | 93.0980 | 0.0000 | 0 | 1760 |
| MLP | 11.9950 | 21.0823 | 0.9080 | 1598 | 162 |
| LSTM | 12.5884 | 21.2210 | 0.9199 | 1619 | 141 |
| PSO-LSTM | 12.4413 | 21.4223 | 0.8915 | 1569 | 191 |
| **XGBoost** | **11.4871** | **20.8878** | **0.9199** | 1619 | 141 |


**Trumpa interpretacija:**

- **XGBoost** pasirodė geriausiai.
- **Seasonal Mean** šiame teste neaptinka pavojingų reikšmių (`Recall = 0`), nes prognozė remiasi istoriniu tos pačios valandos vidurkiu ir neatsižvelgia į dabartinius staigius PM2.5 pokyčius.

**Grafikai** (generuojami notebook 13 sekcijoje, `figures/`):

![Tikra vs prognozuota PM2.5, paskutinės 14 test parų](figures/forecast_vs_actual_14d.png)

![MAE / RMSE / Dangerous Recall palyginimas visiems metodams](figures/metrics_comparison.png)

![Didžiausias test periodo pavojingas pikas su visų metodų prognozėmis](figures/dangerous_peak_episode.png)

