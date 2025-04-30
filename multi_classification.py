import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ссылка на данные
# https://drive.google.com/file/d/1wTkOoA222guACzvIJxLf4wU77Rykp1__/view?usp=sharing
df = pd.read_parquet('./data/B_vsx_vsx.parquet')
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (16, 12)

def classify_type(vtype: str) -> str:
    """
    Возвращает укрупнённый класс переменной звезды
    в зависимости от содержимого строки vtype.
    """

    if pd.isna(vtype) or not isinstance(vtype, str) or vtype.strip() == "":
        return "UNKNOWN"

    # Приведём к верхнему регистру для надёжного поиска подстрок
    t = vtype.upper()

    # --- 1) Затменные (Eclipsing Binaries) ---
    ecl_markers = ["EA", "EB", "EW", "EC", "ELL", "E/RS", "E|", "E "]
    if any(m in t for m in ecl_markers):
        return "ECLIPSING"

    # --- 2) Цефеиды и родственные (DCEP, CW, RV Tauri, ACEP) - пульсирующие
    cep_markers = ["DCEP", "CW-FU", "CW", "CWA", "CWB", "RVA", "RV", "ACEP", "CEP"]
    if any(m in t for m in cep_markers):
        return "PULSATING"

    # --- 3) RR Лиры (RRAB, RRC, RRD, RR...) - пульсирующие
    rr_markers = ["RRAB", "RRC", "RRD", "RR"]
    if any(m in t for m in rr_markers):
        return "PULSATING"

    # --- 4) Короткопериодические пульсаторы: DSCT, SXPHE, GDOR, roAp - пульсирующие
    short_puls = ["DSCT", "HADS", "SXPHE", "GDOR", "ROAP", "ROAM"]
    if any(m in t for m in short_puls):
        return "PULSATING"

    # --- 5) Долгопериодические и полуправильные (M, SR, L) - пульсирующие
    lpv_markers = ["MIRA", "SR", "SRA", "SRB", "SRC", "SRD", "L ", "LB", "LC", "LPV"]
    if any(m in t for m in lpv_markers):
        return "PULSATING"

    # --- 6) Ротационные переменные (BY, RS, ACV, SPB, ROT, GCAS) ---
    rot_markers = ["BY", "RS", "ACV", "SPB", "ROT", "GCAS"]
    if any(m in t for m in rot_markers):
        return "ROTATING"

    # --- 7) Эруптивные/молодые звёзды (T Tauri, EXOR, UXOR, INS...) ---
    yso_markers = ["TTS", "EXOR", "UXOR", "INS", "IN", "INST", "CST", "DYP", "FSCM", "FUOR", "YSO"]
    if any(m in t for m in yso_markers):
        return "ERUPTIVE"

    # --- 8) Катаклизмические (UG, NL, AM, ZAND, IB, IS, ... ) ---
    cataclysmic_markers = ["UG", "NL", "AM", "ZAND", "IB", "ISB", "BE", "DPV", "EXOR", "FUOR", "PNB"]  # и др.
    if any(m in t for m in cataclysmic_markers):
        return "ERUPTIVE"

    if (t == 'E'):
        return "ECLIPSING"

    if (t == 'L'):
        return "PULSATING"

    # если ничего не подошло
    return "UNKNOWN"


# целевая переменная:
df["class"] = df["Type"].apply(classify_type)

# в датасете очень мало эруптивных звезд
print(df["class"].value_counts())
df = df[df['class'] != 'UNKNOWN']
df.drop('Type', inplace=True, axis=1)

df.info()

# plt.figure()
# corr_matrix = df.corr()
# sns.heatmap(corr_matrix, annot=True)
# plt.show()