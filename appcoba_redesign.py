from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. KONFIGURASI WEB
# =========================================================

st.set_page_config(
    page_title="Statistik Pembangunan Manusia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

FILE_DATA = Path(__file__).parent / "Data_Rata_Rata_Rapi.xlsx"

TAHUN = [2010, 2011, 2012, 2013]

KOLOM_IPM = [
    f"IPM_{jk}_{tahun}"
    for tahun in TAHUN
    for jk in ["L", "P"]
]

KOLOM_RLS = [f"RLS_{tahun}" for tahun in TAHUN]
KOLOM_NUMERIK = KOLOM_IPM + KOLOM_RLS


st.markdown("""
<style>
    :root {
        --ink: #172B4D;
        --muted: #667085;
        --line: #E6EAF0;
        --surface: #FFFFFF;
        --canvas: #F6F8FB;
        --blue: #4263EB;
        --teal: #12A594;
        --violet: #8B5CF6;
        --amber: #D99024;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    .block-container { max-width: 1500px; padding: 2rem 2.2rem 3rem; }
    h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -0.025em; }
    h1 { font-weight: 750 !important; }
    h2, h3 { font-weight: 680 !important; }
    p, li, [data-testid="stCaptionContainer"] { color: #475467; }
    [data-testid="stSidebar"] { background: #111C32; border-right: 1px solid #25334D; }
    [data-testid="stSidebar"] * { color: #E8EDF7; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #A9B6CC !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { color: #FFFFFF !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label { font-weight: 550; }
    [data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.25rem; }
    [data-testid="stSidebar"] div[role="radiogroup"] label { padding: 0.35rem 0.45rem; border-radius: 8px; }
    .hero {
        background: linear-gradient(115deg, #172B4D 0%, #243F70 58%, #315B92 100%);
        padding: 30px 32px; border-radius: 16px; color: white; margin: 0 0 18px 0;
        box-shadow: 0 10px 28px rgba(23,43,77,.10); border: 1px solid rgba(255,255,255,.08);
    }
    .hero h1 { color: #FFFFFF !important; font-size: clamp(1.7rem, 2.6vw, 2.25rem); margin: 0 0 8px 0; }
    .hero p { color: #DCE6F7 !important; font-size: 1rem; margin: 0; max-width: 850px; }
    div[data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
        padding: 17px 18px; box-shadow: 0 3px 10px rgba(16,24,40,.035); min-height: 112px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .86rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 720; letter-spacing: -.03em; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: .8rem; }
    .insight {
        background: #FFFFFF; border: 1px solid var(--line); border-left: 3px solid var(--blue);
        border-radius: 10px; padding: 16px 18px; margin: 8px 0 18px; color: var(--ink) !important;
        box-shadow: 0 3px 10px rgba(16,24,40,.025); line-height: 1.6;
    }
    .insight-title { font-weight: 700; color: var(--ink) !important; margin-bottom: 5px; }
    .insight div { color: #475467 !important; }
    .model-header { background: #FFFFFF; border: 1px solid var(--line); border-radius: 12px; padding: 21px 24px; margin: 0 0 18px; box-shadow: 0 3px 10px rgba(16,24,40,.025); }
    .model-header h2 { margin: 9px 0 5px; padding: 0; }
    .model-header p { margin: 0; color: var(--muted) !important; }
    .badge { display: inline-block; background: #EEF2FF; color: #3B55C5; font-size: 11px; font-weight: 750; letter-spacing: .07em; padding: 5px 10px; border-radius: 999px; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    [data-testid="stPlotlyChart"], [data-testid="stPyplot"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 12px; padding: 8px; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 5px; border-bottom: 1px solid var(--line); }
    [data-testid="stTabs"] [data-baseweb="tab"] { padding: 10px 15px; color: #667085; border-radius: 8px 8px 0 0; }
    [data-testid="stTabs"] [aria-selected="true"] { color: var(--blue) !important; font-weight: 700; background: #FFFFFF; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--blue); }
    .stButton button, .stDownloadButton button { border-radius: 8px; font-weight: 650; transition: all .15s ease; }
    .stDownloadButton button { background: #244A86; color: white; border: 1px solid #244A86; }
    .stDownloadButton button:hover { background: #193967; color: white; border-color: #193967; }
    div[data-baseweb="select"] > div, .stTextInput input { border-radius: 8px; }
    hr { border-color: var(--line); }
    @media (max-width: 900px) { .block-container { padding: 1.25rem 1rem 2rem; } .hero { padding: 22px; } }
</style>
""", unsafe_allow_html=True)

# Palet visualisasi konsisten untuk seluruh grafik Matplotlib.
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#D8DEE8",
    "axes.labelcolor": "#475467",
    "axes.titlecolor": "#172B4D",
    "xtick.color": "#667085",
    "ytick.color": "#667085",
    "text.color": "#172B4D",
    "grid.color": "#E9EDF4",
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9.5,
    "legend.frameon": False,
    "lines.linewidth": 2.2,
    "axes.prop_cycle": plt.cycler(color=["#4263EB", "#12A594", "#8B5CF6", "#D99024", "#E25563"]),
})

# =========================================================
# 2. MEMBACA DAN MEMBERSIHKAN DATA
# =========================================================

@st.cache_data
def baca_data(lokasi, waktu_modifikasi):
    df = pd.read_excel(lokasi, sheet_name="Data")

    df.columns = df.columns.astype(str).str.strip()

    wajib = ["Provinsi", "Kabupaten/Kota"] + KOLOM_NUMERIK
    kolom_hilang = [c for c in wajib if c not in df.columns]

    if kolom_hilang:
        raise ValueError(
            "Kolom berikut tidak ditemukan: "
            + ", ".join(kolom_hilang)
        )

    df["Provinsi"] = df["Provinsi"].fillna("Tidak diketahui").astype(str)
    df["Kabupaten/Kota"] = (
        df["Kabupaten/Kota"].fillna("Tidak diketahui").astype(str)
    )

    for kolom in KOLOM_NUMERIK:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

    df["Nama_Wilayah"] = (
        df["Kabupaten/Kota"] + ", " + df["Provinsi"]
    )

    return df


if not FILE_DATA.exists():
    st.error(
        "File Excel belum ditemukan. Periksa lokasi file pada "
        "variabel FILE_DATA di dalam app.py."
    )
    st.stop()


try:
    df = baca_data(
        str(FILE_DATA),
        FILE_DATA.stat().st_mtime
    )
except Exception as error:
    st.error(f"Data tidak berhasil dibaca: {error}")
    st.stop()


# Data kabupaten/kota
data_kabupaten = df.copy()

data_kabupaten["Nama_Wilayah"] = (
    data_kabupaten["Kabupaten/Kota"].str.strip()
)

# Ringkasan data tingkat provinsi
data_provinsi = (
    df.groupby("Provinsi", as_index=False)[KOLOM_NUMERIK]
    .mean()
)

data_provinsi["Kabupaten/Kota"] = data_provinsi["Provinsi"]
data_provinsi["Nama_Wilayah"] = (
    data_provinsi["Provinsi"].str.strip()
)


# =========================================================
# 3. FUNGSI STATISTIK DAN NARASI OTOMATIS
# =========================================================

def format_angka(nilai, desimal=2):
    if pd.isna(nilai):
        return "Tidak tersedia"
    return f"{nilai:,.{desimal}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def statistik(kolom, tahun=None):
    data = df[kolom].dropna()

    if data.empty:
        return None

    return {
        "jumlah": len(data),
        "rata_rata": data.mean(),
        "median": data.median(),
        "minimum": data.min(),
        "maksimum": data.max(),
        "std": data.std(),
        "skewness": data.skew()
    }


def wilayah_ekstrem(kolom, tertinggi=True, data=None):
    sumber = df if data is None else data
    valid = sumber.dropna(subset=[kolom])

    if valid.empty:
        return None

    idx = valid[kolom].idxmax() if tertinggi else valid[kolom].idxmin()
    return valid.loc[idx]


def kartu_temuan(judul, isi):
    st.markdown(
        f"""
        <div class="insight">
            <div class="insight-title">{judul}</div>
            <div>{isi}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def header_model(judul, deskripsi):
    st.markdown(
        f"""
        <div class="model-header">
            <span class="badge">HASIL MODEL</span>
            <h2>{judul}</h2>
            <p>{deskripsi}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def tabel_parameter(baris):
    st.dataframe(
        pd.DataFrame(
            baris,
            columns=["Parameter", "Nilai", "Keterangan"]
        ).astype(str),
        use_container_width=True,
        hide_index=True
    )


def grafik_batang(data, kolom, judul, label_y, jumlah=10):
    valid = data.dropna(subset=[kolom]).nlargest(jumlah, kolom)

    if valid.empty:
        st.info("Belum ada data yang dapat ditampilkan.")
        return

    valid = valid.sort_values(kolom)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(valid["Nama_Wilayah"], valid[kolom], color="#4263EB", height=0.68)
    ax.grid(axis="x", alpha=0.7, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(judul, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel(label_y)
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def kolom_ipm(jenis_kelamin, tahun):
    return f"IPM_{jenis_kelamin}_{tahun}"


def data_per_tahun(data, tahun):
    hasil = data[
        [
            "Provinsi", "Kabupaten/Kota", "Nama_Wilayah",
            kolom_ipm("L", tahun),
            kolom_ipm("P", tahun),
            f"RLS_{tahun}"
        ]
    ].copy()

    hasil.columns = [
        "Provinsi", "Kabupaten/Kota", "Nama_Wilayah",
        "IPM_Laki_laki", "IPM_Perempuan", "RLS"
    ]

    hasil["Selisih_IPM"] = (
        hasil["IPM_Laki_laki"] - hasil["IPM_Perempuan"]
    )

    return hasil


# =========================================================
# 4. NORMALISASI DAN PEMODELAN INTERNAL
#    Tidak ditampilkan sebagai menu di web.
# =========================================================

data_asli = df.copy()

# Normalisasi dengan skala yang konsisten antar-tahun

data_normalisasi = df.copy()

# Normalisasi kolom IPM menggunakan MinMaxScaler
scaler_ipm = MinMaxScaler()
data_normalisasi[KOLOM_IPM] = scaler_ipm.fit_transform(
    df[KOLOM_IPM]
)

# Normalisasi kolom RLS menggunakan MinMaxScaler
scaler_rls = MinMaxScaler()
data_normalisasi[KOLOM_RLS] = scaler_rls.fit_transform(
    df[KOLOM_RLS]
)


# Tahun clustering mengikuti pilihan di sidebar
# Pada proses pertama, tahun default adalah 2013
tahun_cluster = st.session_state.get("tahun_cluster", 2013)

kolom_cluster_l = f"IPM_L_{tahun_cluster}"
kolom_cluster_p = f"IPM_P_{tahun_cluster}"
kolom_cluster = [kolom_cluster_l, kolom_cluster_p]

data_cluster = data_normalisasi.dropna(
    subset=kolom_cluster
).copy()

hasil_cluster = None
jumlah_cluster = None
silhouette_terbaik = None
model_terbaik = None
skor_cluster = []

if len(data_cluster) >= 3:
    X_cluster = data_cluster[kolom_cluster]

    skor_cluster = []
    batas_k = min(6, len(data_cluster) - 1)

    for k in range(2, batas_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        label = model.fit_predict(X_cluster)

        if len(set(label)) > 1:
            skor = silhouette_score(X_cluster, label)
            skor_cluster.append((k, skor, model, label))

    if skor_cluster:
        (
            jumlah_cluster,
            silhouette_terbaik,
            model_terbaik,
            label_terbaik
        ) = max(skor_cluster, key=lambda x: x[1])

        # Simpan data dengan nama kolom yang konsisten
        hasil_cluster = df.loc[
            data_cluster.index,
            ["Provinsi", "Kabupaten/Kota", "Nama_Wilayah"]
        ].copy()

        hasil_cluster["IPM_L"] = df.loc[
            data_cluster.index, kolom_cluster_l
        ]

        hasil_cluster["IPM_P"] = df.loc[
            data_cluster.index, kolom_cluster_p
        ]

        hasil_cluster["Cluster_Asli"] = label_terbaik

        hasil_cluster["Selisih_IPM"] = (
            hasil_cluster["IPM_L"] - hasil_cluster["IPM_P"]
        )

        # Urutkan cluster berdasarkan rata-rata IPM
        profil_cluster = hasil_cluster.groupby(
            "Cluster_Asli"
        ).agg(
            rata_ipm_l=("IPM_L", "mean"),
            rata_ipm_p=("IPM_P", "mean")
        )

        profil_cluster["rata_gabungan"] = (
            profil_cluster["rata_ipm_l"]
            + profil_cluster["rata_ipm_p"]
        ) / 2

        urutan_cluster = profil_cluster[
            "rata_gabungan"
        ].sort_values().index.tolist()

        label_kategori = {
            2: [
                "IPM Relatif Rendah",
                "IPM Relatif Tinggi"
            ],
            3: [
                "IPM Relatif Rendah",
                "IPM Relatif Menengah",
                "IPM Relatif Tinggi"
            ],
            4: [
                "IPM Relatif Rendah",
                "IPM Cenderung Rendah",
                "IPM Cenderung Tinggi",
                "IPM Relatif Tinggi"
            ],
            5: [
                "IPM Relatif Terendah",
                "IPM Cenderung Rendah",
                "IPM Menengah",
                "IPM Cenderung Tinggi",
                "IPM Relatif Tertinggi"
            ]
        }

        if jumlah_cluster in label_kategori:
            daftar_label = label_kategori[jumlah_cluster]
        else:
            daftar_label = [
                f"Kategori IPM urutan ke-{i + 1}"
                for i in range(jumlah_cluster)
            ]

        nama_cluster = {
            cluster: daftar_label[i]
            for i, cluster in enumerate(urutan_cluster)
        }

        hasil_cluster["Karakteristik"] = (
            hasil_cluster["Cluster_Asli"].map(nama_cluster)
        )


# Regresi internal: hubungan RLS dengan IPM
hasil_regresi = {}

for jk, nama in [("L", "Laki-laki"), ("P", "Perempuan")]:
    bagian = []

    for tahun in TAHUN:
        temp = df[
            [
                "Provinsi", "Kabupaten/Kota", "Nama_Wilayah",
                f"RLS_{tahun}", f"IPM_{jk}_{tahun}"
            ]
        ].copy()

        temp.columns = [
            "Provinsi", "Kabupaten_Kota", "Nama_Wilayah",
            "RLS", "IPM"
        ]
        temp["Tahun"] = tahun
        bagian.append(temp)

    data_model = pd.concat(bagian, ignore_index=True)
    data_model = data_model.dropna(subset=["RLS", "IPM"])

    if len(data_model) >= 5 and data_model["Nama_Wilayah"].nunique() >= 2:
        X = data_model[["RLS"]]
        y = data_model["IPM"]
        groups = data_model["Nama_Wilayah"]

        pembagi = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=42
        )

        indeks_latih, indeks_uji = next(
            pembagi.split(X, y, groups=groups)
        )

        model = LinearRegression()
        model.fit(X.iloc[indeks_latih], y.iloc[indeks_latih])

        prediksi = model.predict(X.iloc[indeks_uji])
        aktual = y.iloc[indeks_uji]

        hasil_regresi[nama] = {
            "model": model,
            "aktual": aktual,
            "prediksi": prediksi,
            "mae": mean_absolute_error(aktual, prediksi),
            "rmse": np.sqrt(mean_squared_error(aktual, prediksi)),
            "r2": r2_score(aktual, prediksi),
            "data_uji": data_model.iloc[indeks_uji].copy(),
            "data_latih": data_model.iloc[indeks_latih].copy(),
            "n_total": len(data_model),
            "n_wilayah": data_model["Nama_Wilayah"].nunique(),
            "params_split": {
                "n_splits": pembagi.n_splits,
                "test_size": pembagi.test_size,
                "random_state": pembagi.random_state
            }
        }


# =========================================================
# 5. SIDEBAR DAN FILTER
# =========================================================

st.sidebar.markdown("## ◈ Human Development")
st.sidebar.caption("IPM · RLS · Analisis Wilayah")
st.sidebar.markdown("---")
st.sidebar.caption("NAVIGASI")

BAGIAN_HALAMAN = {
    "📈 Statistik Umum": [
        "Ringkasan",
        "Statistik IPM",
        "Statistik Pendidikan",
        "Perbandingan Wilayah",
        "Kategori RLS & Korelasi"
    ],
    "🤖 Hasil Model": [
        "K-Means Clustering",
        "Regresi Linear"
    ],
    "📘 Tentang Data": ["Tentang Data"]
}

bagian = st.sidebar.radio("Bagian", list(BAGIAN_HALAMAN))

if len(BAGIAN_HALAMAN[bagian]) > 1:
    halaman = st.sidebar.radio(
        "Halaman",
        BAGIAN_HALAMAN[bagian],
        key=f"halaman_{bagian}"
    )
else:
    halaman = BAGIAN_HALAMAN[bagian][0]

st.sidebar.markdown("---")
st.sidebar.caption("FILTER DATA")
tahun_pilihan = st.sidebar.selectbox(
    "Tahun analisis",
    TAHUN,
    index=len(TAHUN) - 1,
    key="tahun_cluster"
)

provinsi_pilihan = st.sidebar.selectbox(
    "Provinsi",
    ["Semua Provinsi"] + sorted(df["Provinsi"].dropna().astype(str).unique().tolist()),
    key="filter_provinsi"
)

if provinsi_pilihan == "Semua Provinsi":
    pilihan_wilayah = sorted(df["Nama_Wilayah"].dropna().astype(str).unique().tolist())
else:
    pilihan_wilayah = sorted(
        df.loc[df["Provinsi"] == provinsi_pilihan, "Nama_Wilayah"]
        .dropna().astype(str).unique().tolist()
    )

wilayah_pilihan = st.sidebar.selectbox(
    "Kabupaten/kota",
    ["Semua Kabupaten/Kota"] + pilihan_wilayah,
    key="filter_wilayah"
)

data_tampilan = df.copy()
if provinsi_pilihan != "Semua Provinsi":
    data_tampilan = data_tampilan[data_tampilan["Provinsi"] == provinsi_pilihan]
if wilayah_pilihan != "Semua Kabupaten/Kota":
    data_tampilan = data_tampilan[data_tampilan["Nama_Wilayah"] == wilayah_pilihan]

data_tampilan = data_tampilan.copy()

data_tahun = data_per_tahun(data_tampilan, tahun_pilihan)

kolom_l = f"IPM_L_{tahun_pilihan}"
kolom_p = f"IPM_P_{tahun_pilihan}"
kolom_rls = f"RLS_{tahun_pilihan}"


# =========================================================
# 6. HEADER
# =========================================================

st.markdown(
    f"""
    <div class="hero">
        <h1>Statistik Pembangunan Manusia dan Pendidikan</h1>
        <p>
            Menjelajahi perbedaan IPM, capaian pendidikan,
            dan karakteristik wilayah berdasarkan data BPS.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    f"Periode data 2010–2013  ·  Tahun analisis: {tahun_pilihan} "
    f"·  Provinsi: {provinsi_pilihan}  ·  Wilayah: {wilayah_pilihan}"
)

# =========================================================
# 7. HALAMAN RINGKASAN
# =========================================================

if halaman == "Ringkasan":
    st.subheader("Gambaran Umum")

    rata_l = data_tampilan[kolom_l].mean()
    rata_p = data_tampilan[kolom_p].mean()
    rata_rls = data_tampilan[kolom_rls].mean()
    jumlah_wilayah = data_tampilan["Nama_Wilayah"].nunique()

    selisih_rata = rata_l - rata_p

    a, b, c, d = st.columns(4)

    a.metric("Wilayah", f"{jumlah_wilayah:,}")
    b.metric("Rata-rata IPM laki-laki", format_angka(rata_l))
    c.metric("Rata-rata IPM perempuan", format_angka(rata_p))
    d.markdown(
    f"""
    <div style="
        background-color: white;
        border: 1px solid #E1E8E3;
        padding: 16px;
        border-radius: 14px;
    ">
        <div style="font-size: 14px; color: #183D32;">
            Rata-rata lama sekolah
        </div>
        <div style="color: #183D32; margin-top: 8px;">
            <span style="font-size: 28px; font-weight: 600;">
                {format_angka(rata_rls)}
            </span>
            <span style="font-size: 12px !important; font-weight: 400;">
                tahun
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
    st.subheader("Sorotan Statistik")

    if pd.notna(selisih_rata):
        if selisih_rata > 0:
            narasi = (
                f"Pada {tahun_pilihan}, rata-rata IPM laki-laki lebih tinggi "
                f"daripada IPM perempuan sebesar {format_angka(selisih_rata)} poin "
                f"pada wilayah yang sedang ditampilkan."
            )
        elif selisih_rata < 0:
            narasi = (
                f"Pada {tahun_pilihan}, rata-rata IPM perempuan lebih tinggi "
                f"daripada IPM laki-laki sebesar {format_angka(abs(selisih_rata))} poin "
                f"pada wilayah yang sedang ditampilkan."
            )
        else:
            narasi = (
                f"Pada {tahun_pilihan}, rata-rata IPM laki-laki dan perempuan "
                "memiliki nilai yang sama pada wilayah yang sedang ditampilkan."
            )
        kartu_temuan("Perbandingan IPM berdasarkan jenis kelamin", narasi)

    tertinggi_l = wilayah_ekstrem(kolom_l, True, data_tampilan)
    terendah_l = wilayah_ekstrem(kolom_l, False, data_tampilan)

    if tertinggi_l is not None and terendah_l is not None:
        jarak = tertinggi_l[kolom_l] - terendah_l[kolom_l]
        kartu_temuan(
            "Rentang capaian IPM laki-laki",
            f"{tertinggi_l['Nama_Wilayah']} memiliki nilai tertinggi "
            f"({format_angka(tertinggi_l[kolom_l])}), sedangkan "
            f"{terendah_l['Nama_Wilayah']} memiliki nilai terendah "
            f"({format_angka(terendah_l[kolom_l])}). "
            f"Rentang keduanya adalah {format_angka(jarak)} poin."
        )

    tertinggi_rls = wilayah_ekstrem(kolom_rls, True, data_tampilan)
    terendah_rls = wilayah_ekstrem(kolom_rls, False, data_tampilan)

    if tertinggi_rls is not None and terendah_rls is not None:
        rentang_rls = tertinggi_rls[kolom_rls] - terendah_rls[kolom_rls]
        kartu_temuan(
            "Perbedaan capaian pendidikan",
            f"{tertinggi_rls['Nama_Wilayah']} mencatat rata-rata lama sekolah "
            f"tertinggi sebesar {format_angka(tertinggi_rls[kolom_rls])} tahun, "
            f"sedangkan {terendah_rls['Nama_Wilayah']} mencatat "
            f"{format_angka(terendah_rls[kolom_rls])} tahun. "
            f"Selisihnya {format_angka(rentang_rls)} tahun."
        )

    kiri, kanan = st.columns(2)

    with kiri:
        grafik_batang(
            data_tampilan,
            kolom_l,
            f"10 Wilayah dengan IPM Laki-laki Tertinggi ({tahun_pilihan})",
            "Nilai IPM"
        )

    with kanan:
        grafik_batang(
            data_tampilan,
            kolom_rls,
            f"10 Wilayah dengan Rata-rata Lama Sekolah Tertinggi ({tahun_pilihan})",
            "Tahun"
        )

    st.subheader("Hubungan Pendidikan dan IPM")

    pasangan = data_tampilan[[kolom_rls, kolom_l, kolom_p]].dropna()

    if len(pasangan) >= 3:
        corr_l = pasangan[kolom_rls].corr(pasangan[kolom_l])
        corr_p = pasangan[kolom_rls].corr(pasangan[kolom_p])

        x, y = st.columns(2)
        x.metric("Korelasi RLS–IPM laki-laki", format_angka(corr_l))
        y.metric("Korelasi RLS–IPM perempuan", format_angka(corr_p))

        if pd.notna(corr_l):
            if corr_l >= 0.7:
                deskripsi_l = "hubungan positif yang kuat"
            elif corr_l >= 0.3:
                deskripsi_l = "hubungan positif sedang"
            elif corr_l > 0:
                deskripsi_l = "hubungan positif yang lemah"
            elif corr_l <= -0.7:
                deskripsi_l = "hubungan negatif yang kuat"
            elif corr_l <= -0.3:
                deskripsi_l = "hubungan negatif sedang"
            else:
                deskripsi_l = "hubungan yang lemah atau mendekati nol"

            kartu_temuan(
                "Pola hubungan pendidikan dan IPM",
                f"Pada {tahun_pilihan}, RLS dan IPM laki-laki menunjukkan "
                f"{deskripsi_l} berdasarkan korelasi Pearson "
                f"sebesar {format_angka(corr_l)}. Korelasi menunjukkan hubungan, "
                "bukan bukti sebab-akibat."
            )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(pasangan[kolom_rls], pasangan[kolom_l], alpha=0.65)
        ax.set_xlabel("Rata-rata lama sekolah (tahun)")
        ax.set_ylabel("IPM laki-laki")
        ax.set_title("RLS dan IPM Laki-laki")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Data valid belum cukup untuk menghitung korelasi.")

    
    st.caption("Analisis K-Means dan evaluasi regresi tersedia pada menu Hasil Model di sidebar.")


# =========================================================
# 8. HALAMAN STATISTIK IPM
# =========================================================

elif halaman == "Statistik IPM":
    st.subheader("Statistik Indeks Pembangunan Manusia")

    kiri, kanan = st.columns(2)

    for kolom, nama, wadah in [
        (kolom_l, "IPM Laki-laki", kiri),
        (kolom_p, "IPM Perempuan", kanan)
    ]:
        with wadah:
            nilai = data_tampilan[kolom].dropna()

            st.markdown(f"### {nama}")

            if not nilai.empty:
                m1, m2 = st.columns(2)
                m1.metric("Rata-rata", format_angka(nilai.mean()))
                m2.metric("Median", format_angka(nilai.median()))

                m3, m4 = st.columns(2)
                m3.metric("Terendah", format_angka(nilai.min()))
                m4.metric("Tertinggi", format_angka(nilai.max()))

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(nilai, bins=15, edgecolor="white")
                ax.set_xlabel("Nilai IPM")
                ax.set_ylabel("Jumlah wilayah")
                ax.set_title(f"Distribusi {nama}")
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(6, 2.5))
                ax.boxplot(nilai, vert=False)
                ax.set_xlabel("Nilai IPM")
                ax.set_title(f"Sebaran {nama}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.subheader("Selisih IPM Antarjenis Kelamin")

    valid = data_tahun.dropna(
        subset=["IPM_Laki_laki", "IPM_Perempuan"]
    ).copy()

    if not valid.empty:
        valid["Selisih_Absolut"] = valid["Selisih_IPM"].abs()
        valid = valid.sort_values("Selisih_Absolut", ascending=False)

        wilayah_besar = valid.iloc[0]
        wilayah_kecil = valid.iloc[-1]

        kartu_temuan(
            "Perbedaan IPM terbesar",
            f"{wilayah_besar['Nama_Wilayah']} memiliki selisih absolut IPM "
            f"terbesar sebesar {format_angka(wilayah_besar['Selisih_Absolut'])} poin "
            f"pada {tahun_pilihan}. Nilai positif berarti IPM laki-laki lebih tinggi; "
            "nilai negatif pada selisih bertanda menunjukkan IPM perempuan lebih tinggi."
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        tampil = valid.nlargest(10, "Selisih_Absolut").sort_values(
            "Selisih_IPM"
        )
        ax.barh(tampil["Nama_Wilayah"], tampil["Selisih_IPM"])
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("IPM laki-laki dikurangi IPM perempuan")
        ax.set_title("10 Wilayah dengan Selisih IPM Absolut Terbesar")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            valid[
                [
                    "Provinsi", "Kabupaten/Kota",
                    "IPM_Laki_laki", "IPM_Perempuan", "Selisih_IPM"
                ]
            ].rename(columns={
                "IPM_Laki_laki": "IPM Laki-laki",
                "IPM_Perempuan": "IPM Perempuan",
                "Selisih_IPM": "Selisih IPM"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Perkembangan IPM 2010–2013")

    tren = []
    for tahun in TAHUN:
        tren.append({
            "Tahun": tahun,
            "IPM Laki-laki": data_tampilan[f"IPM_L_{tahun}"].mean(),
            "IPM Perempuan": data_tampilan[f"IPM_P_{tahun}"].mean()
        })

    
    df_tren = pd.DataFrame(tren)
    df_tren["Tahun"] = df_tren["Tahun"].astype(str)
    df_tren = df_tren.set_index("Tahun")

    st.line_chart(df_tren)

    if df_tren[["IPM Laki-laki", "IPM Perempuan"]].notna().any().any():
        for nama_kolom in ["IPM Laki-laki", "IPM Perempuan"]:
            seri = df_tren[nama_kolom].dropna()
            if len(seri) >= 2:
                perubahan = seri.iloc[-1] - seri.iloc[0]
                arah = "meningkat" if perubahan > 0 else (
                    "menurun" if perubahan < 0 else "tidak berubah"
                )
                kartu_temuan(
                    f"Tren {nama_kolom}",
                    f"Rata-rata {nama_kolom.lower()} {arah} sebesar "
                    f"{format_angka(abs(perubahan))} poin dari "
                    f"{seri.index[0]} hingga {seri.index[-1]}."
                )



# =========================================================
# 9. HALAMAN STATISTIK PENDIDIKAN
# =========================================================

elif halaman == "Statistik Pendidikan":
    st.subheader("Statistik Rata-rata Lama Sekolah")

    nilai = data_tampilan[kolom_rls].dropna()

    if nilai.empty:
        st.warning("Tidak ada data RLS yang tersedia untuk pilihan ini.")
    else:
        # Kartu statistik dengan ukuran tulisan "tahun" lebih kecil
        a, b, c, d = st.columns(4)

        statistik_rls = [
            (a, "Rata-rata", nilai.mean()),
            (b, "Median", nilai.median()),
            (c, "Nilai terendah", nilai.min()),
            (d, "Nilai tertinggi", nilai.max())
        ]

        for kolom, label, angka in statistik_rls:
            with kolom:
                st.markdown(
                    f"""
                    <div style="
                        background-color: #FFFFFF;
                        border: 1px solid #E1E8E3;
                        padding: 16px;
                        border-radius: 14px;
                        min-height: 100px;
                    ">
                        <div style="
                            font-size: 14px;
                            color: #52645A;
                            margin-bottom: 8px;
                        ">
                            {label}
                        </div>
                        <div style="
                            color: #183D32;
                            white-space: nowrap;
                        ">
                            <span style="
                                font-size: 28px;
                                font-weight: 600;
                            ">
                                {format_angka(angka)}
                            </span>
                            <span style="
                                font-size: 12px !important;
                                font-weight: 400;
                            ">
                                tahun
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Temuan wilayah dengan RLS tertinggi dan terendah
        rendah = wilayah_ekstrem(kolom_rls, False, data_tampilan)
        tinggi = wilayah_ekstrem(kolom_rls, True, data_tampilan)

        if rendah is not None and tinggi is not None:
            kartu_temuan(
                "Kesenjangan capaian pendidikan",
                f"Pada {tahun_pilihan}, {tinggi['Nama_Wilayah']} memiliki "
                f"rata-rata lama sekolah tertinggi sebesar "
                f"{format_angka(tinggi[kolom_rls])} tahun. "
                f"Nilai terendah terdapat di {rendah['Nama_Wilayah']}, "
                f"yaitu {format_angka(rendah[kolom_rls])} tahun. "
                f"Rentang antara keduanya adalah "
                f"{format_angka(tinggi[kolom_rls] - rendah[kolom_rls])} tahun."
            )

        # Grafik distribusi dan sebaran data
        kiri, kanan = st.columns(2)

        with kiri:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(nilai, bins=15, edgecolor="white")
            ax.set_title("Distribusi Rata-rata Lama Sekolah")
            ax.set_xlabel("Tahun sekolah")
            ax.set_ylabel("Jumlah wilayah")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with kanan:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.boxplot(nilai, vert=False)
            ax.set_title("Sebaran Rata-rata Lama Sekolah")
            ax.set_xlabel("Tahun sekolah")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Grafik 10 wilayah dengan RLS tertinggi
        st.subheader("Wilayah dengan RLS Tertinggi")

        grafik_batang(
            data_tampilan,
            kolom_rls,
            f"10 Wilayah Teratas pada {tahun_pilihan}",
            "Rata-rata lama sekolah (tahun)"
        )

        # Perkembangan rata-rata lama sekolah dari 2010–2013
        st.subheader("Perkembangan Pendidikan")

        
        tren_rls = pd.DataFrame({
        "Tahun": [str(tahun) for tahun in TAHUN],
        "Rata-rata lama sekolah": [
        data_tampilan[f"RLS_{tahun}"].mean()
        for tahun in TAHUN
        ]}).set_index("Tahun")

        st.line_chart(tren_rls)

        awal = tren_rls["Rata-rata lama sekolah"].dropna()

        if len(awal) >= 2:
            perubahan = awal.iloc[-1] - awal.iloc[0]

            arah = (
                "meningkat" if perubahan > 0
                else "menurun" if perubahan < 0
                else "tidak berubah"
            )

            kartu_temuan(
                "Perubahan rata-rata lama sekolah",
                f"Rata-rata lama sekolah {arah} sebesar "
                f"{format_angka(abs(perubahan))} tahun dari "
                f"{awal.index[0]} hingga {awal.index[-1]}."
            )




# =========================================================
# 10. HALAMAN PERBANDINGAN WILAYAH
# =========================================================

elif halaman == "Perbandingan Wilayah":
    st.subheader("Bandingkan Dua Wilayah")

    st.caption(
        "Pilih provinsi dan kabupaten/kota untuk kedua wilayah "
        "yang ingin dibandingkan."
    )

    daftar_provinsi = sorted(
        data_kabupaten["Provinsi"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    # Pilihan dua wilayah
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Wilayah Pertama")

        provinsi_a = st.selectbox(
            "Pilih provinsi pertama",
            daftar_provinsi,
            key="provinsi_a"
        )

        daftar_kab_a = sorted(
            data_kabupaten.loc[
                data_kabupaten["Provinsi"].str.strip() == provinsi_a,
                "Kabupaten/Kota"
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        wilayah_a = st.selectbox(
            "Pilih kabupaten/kota pertama",
            daftar_kab_a,
            key="wilayah_a"
        )

        baris_a = data_kabupaten[
            (data_kabupaten["Provinsi"].str.strip() == provinsi_a)
            & (
                data_kabupaten["Kabupaten/Kota"].str.strip()
                == wilayah_a
            )
        ].iloc[0]

    with col2:
        st.markdown("### Wilayah Kedua")

        provinsi_b = st.selectbox(
            "Pilih provinsi kedua",
            daftar_provinsi,
            key="provinsi_b"
        )

        daftar_kab_b = sorted(
            data_kabupaten.loc[
                data_kabupaten["Provinsi"].str.strip() == provinsi_b,
                "Kabupaten/Kota"
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        wilayah_b = st.selectbox(
            "Pilih kabupaten/kota kedua",
            daftar_kab_b,
            key="wilayah_b"
        )

        baris_b = data_kabupaten[
            (data_kabupaten["Provinsi"].str.strip() == provinsi_b)
            & (
                data_kabupaten["Kabupaten/Kota"].str.strip()
                == wilayah_b
            )
        ].iloc[0]

    nama_a = f"{wilayah_a}, {provinsi_a}"
    nama_b = f"{wilayah_b}, {provinsi_b}"

    # Kolom sesuai tahun yang dipilih di sidebar
    kolom_l = f"IPM_L_{tahun_pilihan}"
    kolom_p = f"IPM_P_{tahun_pilihan}"
    kolom_rls = f"RLS_{tahun_pilihan}"

    st.markdown("---")
    st.markdown(f"### Perbandingan Tahun {tahun_pilihan}")

    perbandingan = pd.DataFrame({
        "Indikator": [
            "IPM laki-laki",
            "IPM perempuan",
            "Rata-rata lama sekolah"
        ],
        nama_a: [
            baris_a[kolom_l],
            baris_a[kolom_p],
            baris_a[kolom_rls]
        ],
        nama_b: [
            baris_b[kolom_l],
            baris_b[kolom_p],
            baris_b[kolom_rls]
        ]
    })

    st.dataframe(
        perbandingan,
        use_container_width=True,
        hide_index=True
    )

    
    # Grafik gabungan tren IPM laki-laki dan perempuan
    st.subheader("Tren IPM 2010–2013")

    tren_ipm = pd.DataFrame({
        "Tahun": [str(tahun) for tahun in TAHUN],
        f"IPM Laki-laki — {nama_a}": [
            baris_a[f"IPM_L_{tahun}"] for tahun in TAHUN
        ],
        f"IPM Perempuan — {nama_a}": [
            baris_a[f"IPM_P_{tahun}"] for tahun in TAHUN
        ],
        f"IPM Laki-laki — {nama_b}": [
            baris_b[f"IPM_L_{tahun}"] for tahun in TAHUN
        ],
        f"IPM Perempuan — {nama_b}": [
            baris_b[f"IPM_P_{tahun}"] for tahun in TAHUN
        ]
    }).set_index("Tahun")

    st.line_chart(tren_ipm)

    # Grafik tren rata-rata lama sekolah tetap terpisah
    st.subheader("Tren Rata-rata Lama Sekolah 2010–2013")

    tren_rls = pd.DataFrame({
        "Tahun": [str(tahun) for tahun in TAHUN],
        nama_a: [
            baris_a[f"RLS_{tahun}"] for tahun in TAHUN
        ],
        nama_b: [
            baris_b[f"RLS_{tahun}"] for tahun in TAHUN
        ]
    }).set_index("Tahun")

    st.line_chart(tren_rls)


# =========================================================
# 11. HALAMAN KATEGORI IPM, RLS, DAN HUBUNGAN KEDUANYA
# =========================================================

elif halaman == "Kategori RLS & Korelasi":
    st.subheader("Kategori RLS dan Hubungannya dengan IPM")

    st.caption(
        "Kategori relatif menunjukkan posisi suatu wilayah dibandingkan "
        "wilayah lain dalam cakupan data yang ditampilkan."
    )

    # -----------------------------------------------------
    # B. KATEGORI RLS BERDASARKAN TAHUN PILIHAN
    # -----------------------------------------------------
    st.markdown("### 1. Kategori Wilayah Berdasarkan RLS")

    st.caption(
        f"RLS tahun {tahun_pilihan} dibagi menjadi tiga kelompok relatif "
        "berdasarkan peringkat persentil wilayah."
    )

    data_rls = data_tampilan[
        ["Provinsi", "Kabupaten/Kota", kolom_rls]
    ].dropna(subset=[kolom_rls]).copy()

    if not data_rls.empty:
        persentil = data_rls[kolom_rls].rank(
            pct=True,
            method="average"
        )

        data_rls["Kategori_RLS"] = pd.cut(
            persentil,
            bins=[0, 1/3, 2/3, 1],
            labels=[
                "RLS Relatif Rendah",
                "RLS Relatif Menengah",
                "RLS Relatif Tinggi"
            ],
            include_lowest=True
        )

        urutan_rls = [
            "RLS Relatif Rendah",
            "RLS Relatif Menengah",
            "RLS Relatif Tinggi"
        ]

        for kategori in urutan_rls:
            bagian = data_rls[
                data_rls["Kategori_RLS"] == kategori
            ].copy()

            with st.expander(
                f"{kategori} — {len(bagian)} wilayah"
            ):
                st.metric(
                    "Rata-rata lama sekolah",
                    f"{format_angka(bagian[kolom_rls].mean())} tahun"
                )

                st.write(
                    f"Rentang nilai: "
                    f"{format_angka(bagian[kolom_rls].min())}–"
                    f"{format_angka(bagian[kolom_rls].max())} tahun"
                )

                daftar = bagian[
                    ["Provinsi", "Kabupaten/Kota", kolom_rls]
                ].sort_values(
                    kolom_rls,
                    ascending=True
                )

                daftar.columns = [
                    "Provinsi",
                    "Kabupaten/Kota",
                    f"RLS {tahun_pilihan} (tahun)"
                ]

                st.dataframe(
                    daftar.round(2),
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    f"Unduh daftar {kategori}",
                    data=daftar.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name=(
                        kategori.lower()
                        .replace(" ", "_")
                        + f"_{tahun_pilihan}.csv"
                    ),
                    mime="text/csv",
                    key=f"unduh_rls_{kategori}_{tahun_pilihan}"
                )

    else:
        st.info("Data RLS yang valid belum tersedia.")

    # -----------------------------------------------------
    # C. HUBUNGAN RLS DENGAN IPM TAHUN 2010–2013
    # -----------------------------------------------------
    st.markdown("### 2. Hubungan RLS dengan IPM Tahun 2010–2013")

    st.write(
        "Analisis menggunakan korelasi Pearson untuk mengetahui arah "
        "dan kekuatan hubungan linear antara RLS dan IPM pada setiap tahun."
    )

    hasil_korelasi = []

    for tahun in TAHUN:
        rls = f"RLS_{tahun}"
        ipm_l = f"IPM_L_{tahun}"
        ipm_p = f"IPM_P_{tahun}"

        pasangan_l = data_tampilan[
            [rls, ipm_l]
        ].dropna()

        pasangan_p = data_tampilan[
            [rls, ipm_p]
        ].dropna()

        korelasi_l = (
            pasangan_l[rls].corr(pasangan_l[ipm_l])
            if len(pasangan_l) >= 3
            else np.nan
        )

        korelasi_p = (
            pasangan_p[rls].corr(pasangan_p[ipm_p])
            if len(pasangan_p) >= 3
            else np.nan
        )

        hasil_korelasi.append({
            "Tahun": tahun,
            "Jumlah wilayah IPM laki-laki": len(pasangan_l),
            "Jumlah wilayah IPM perempuan": len(pasangan_p),
            "Korelasi RLS–IPM laki-laki": korelasi_l,
            "Korelasi RLS–IPM perempuan": korelasi_p
        })

    tabel_korelasi = pd.DataFrame(hasil_korelasi)

    st.dataframe(
        tabel_korelasi.round(3),
        use_container_width=True,
        hide_index=True
    )

    
    # -----------------------------------------------------
    # HEATMAP KORELASI RLS DAN IPM
    # -----------------------------------------------------
    st.markdown("#### Heatmap Korelasi RLS dengan IPM")

    st.write(
        "Warna menunjukkan arah dan kekuatan hubungan linear "
        "antara rata-rata lama sekolah (RLS) dan IPM. "
        "Hijau menunjukkan korelasi positif, merah menunjukkan "
        "korelasi negatif, dan warna di sekitar kuning menunjukkan "
        "korelasi yang mendekati nol."
    )

    kolom_heatmap = [
        "Korelasi RLS–IPM laki-laki",
        "Korelasi RLS–IPM perempuan"
    ]

    nilai_heatmap = (
        tabel_korelasi
        .set_index("Tahun")[kolom_heatmap]
        .to_numpy(dtype=float)
    )

    # Warna khusus untuk nilai korelasi yang tidak tersedia
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#E5E7EB")

    fig, ax = plt.subplots(figsize=(9, 4.5))

    gambar = ax.imshow(
        nilai_heatmap,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        "IPM Laki-laki",
        "IPM Perempuan"
    ])

    ax.set_yticks(range(len(tabel_korelasi)))
    ax.set_yticklabels(tabel_korelasi["Tahun"])

    ax.set_xlabel("Indikator IPM")
    ax.set_ylabel("Tahun")
    ax.set_title("Korelasi RLS dengan IPM Tahun 2010–2013")

    # Menampilkan angka korelasi di setiap sel
    for i in range(nilai_heatmap.shape[0]):
        for j in range(nilai_heatmap.shape[1]):
            nilai = nilai_heatmap[i, j]

            if np.isfinite(nilai):
                teks = f"{nilai:.3f}"
            else:
                teks = "N/A"

            ax.text(
                j,
                i,
                teks,
                ha="center",
                va="center",
                color="black",
                fontsize=11
            )

    colorbar = fig.colorbar(
        gambar,
        ax=ax,
        fraction=0.045,
        pad=0.04
    )
    colorbar.set_label("Koefisien korelasi Pearson")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


    # -----------------------------------------------------
    # D. GRAFIK PERUBAHAN KORELASI
    # -----------------------------------------------------
    st.markdown("#### Perubahan Korelasi Antar Tahun")

    grafik_korelasi = tabel_korelasi.set_index("Tahun")[
        [
            "Korelasi RLS–IPM laki-laki",
            "Korelasi RLS–IPM perempuan"
        ]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        grafik_korelasi.index,
        grafik_korelasi["Korelasi RLS–IPM laki-laki"],
        marker="o",
        label="IPM laki-laki"
    )

    ax.plot(
        grafik_korelasi.index,
        grafik_korelasi["Korelasi RLS–IPM perempuan"],
        marker="o",
        label="IPM perempuan"
    )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(TAHUN)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Koefisien korelasi Pearson")
    ax.set_title("Hubungan RLS dengan IPM, 2010–2013")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # -----------------------------------------------------
    # E. GRAFIK SEBAR RLS DAN IPM PADA SETIAP TAHUN
    # -----------------------------------------------------
    st.markdown("#### Pola Sebaran RLS dan IPM")

    tahun_grafik = st.selectbox(
        "Pilih tahun untuk melihat pola sebaran",
        TAHUN,
        index=len(TAHUN) - 1,
        key="tahun_scatter_rls_ipm"
    )

    rls_grafik = f"RLS_{tahun_grafik}"
    ipm_l_grafik = f"IPM_L_{tahun_grafik}"
    ipm_p_grafik = f"IPM_P_{tahun_grafik}"

    data_scatter = data_tampilan[
        [
            "Provinsi",
            "Kabupaten/Kota",
            rls_grafik,
            ipm_l_grafik,
            ipm_p_grafik
        ]
    ].dropna(
        subset=[rls_grafik, ipm_l_grafik, ipm_p_grafik]
    )

    if len(data_scatter) >= 3:
        kiri, kanan = st.columns(2)

        for wadah, kolom_ipm, judul in [
            (kiri, ipm_l_grafik, "IPM laki-laki"),
            (kanan, ipm_p_grafik, "IPM perempuan")
        ]:
            with wadah:
                fig, ax = plt.subplots(figsize=(6, 4))

                ax.scatter(
                    data_scatter[rls_grafik],
                    data_scatter[kolom_ipm],
                    alpha=0.7
                )

                # Garis tren linear untuk membantu membaca arah hubungan
                x = data_scatter[rls_grafik].to_numpy()
                y = data_scatter[kolom_ipm].to_numpy()

                if len(np.unique(x)) >= 2:
                    koefisien = np.polyfit(x, y, 1)
                    garis = np.poly1d(koefisien)
                    x_urut = np.linspace(x.min(), x.max(), 100)

                    ax.plot(
                        x_urut,
                        garis(x_urut),
                        linestyle="--",
                        label="Garis tren"
                    )

                    ax.legend()

                ax.set_xlabel("Rata-rata lama sekolah (tahun)")
                ax.set_ylabel(judul)
                ax.set_title(f"RLS dan {judul} ({tahun_grafik})")
                ax.spines[["top", "right"]].set_visible(False)

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                nilai_corr = data_scatter[
                    rls_grafik
                ].corr(data_scatter[kolom_ipm])

                st.metric(
                    f"Korelasi {judul}",
                    format_angka(nilai_corr, 3)
                )

    else:
        st.info(
            "Data valid belum cukup untuk membuat grafik sebar."
        )

    # -----------------------------------------------------
    # F. RINGKASAN INTERPRETASI
    # -----------------------------------------------------
    st.markdown("### 3. Ringkasan Hasil Analisis")

    def interpretasi_korelasi(nilai):
        if pd.isna(nilai):
            return "belum dapat dihitung"
        elif nilai >= 0.7:
            return "positif kuat"
        elif nilai >= 0.3:
            return "positif sedang"
        elif nilai > 0:
            return "positif lemah"
        elif nilai <= -0.7:
            return "negatif kuat"
        elif nilai <= -0.3:
            return "negatif sedang"
        elif nilai < 0:
            return "negatif lemah"
        return "mendekati nol"

    for kolom, nama in [
        ("Korelasi RLS–IPM laki-laki", "IPM laki-laki"),
        ("Korelasi RLS–IPM perempuan", "IPM perempuan")
    ]:
        nilai_valid = tabel_korelasi[kolom].dropna()

        if not nilai_valid.empty:
            rata_korelasi = nilai_valid.mean()

            kartu_temuan(
                f"Hubungan RLS dengan {nama.lower()}",
                f"Rata-rata koefisien korelasi tahunan selama "
                f"2010–2013 adalah {format_angka(rata_korelasi, 3)}, "
                f"yang menunjukkan hubungan {interpretasi_korelasi(rata_korelasi)}. "
                "Nilai rata-rata ini merupakan ringkasan deskriptif. "
                "Perhatikan pula nilai pada setiap tahun karena kekuatan "
                "hubungan dapat berubah. Korelasi tidak membuktikan "
                "hubungan sebab-akibat."
            )

    st.caption(
        "Catatan: kategori RLS adalah kategori relatif dalam dataset "
        "(berdasarkan persentil), bukan klasifikasi resmi BPS."
    )



# =========================================================
# 11b. HALAMAN HASIL MODEL
# =========================================================

elif halaman == "K-Means Clustering":
    header_model(
        "K-Means Clustering",
        "Mengelompokkan kabupaten/kota yang memiliki pola IPM "
        "laki-laki dan perempuan yang mirip."
    )

    if hasil_cluster is None:
        st.info(
            "Pengelompokan belum tersedia karena data valid "
            "belum mencukupi."
        )
    else:
        st.caption(
            f"Model dilatih pada seluruh wilayah dengan data IPM tahun "
            f"{tahun_cluster} (mengikuti pilihan tahun di sidebar). "
            "Filter provinsi hanya menyaring tampilan hasil, bukan "
            "data latih model."
        )

        tab_konfig, tab_hasil, tab_eval, tab_daftar = st.tabs([
            "⚙️ Konfigurasi Model",
            "📊 Hasil Pengelompokan",
            "🎯 Evaluasi Silhouette",
            "📋 Daftar Wilayah"
        ])

        with tab_konfig:
            p = model_terbaik.get_params()
            k_uji = [s[0] for s in skor_cluster]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Algoritma", "K-Means")
            c2.metric("Jumlah cluster (k)", jumlah_cluster)
            c3.metric("Silhouette score", f"{silhouette_terbaik:.4f}")
            c4.metric("Jumlah wilayah", f"{len(data_cluster):,}")

            st.markdown("#### Parameter model")
            tabel_parameter([
                ("n_clusters", p["n_clusters"],
                 "Jumlah cluster, dipilih otomatis dari silhouette score tertinggi"),
                ("n_init", p["n_init"],
                 "Jumlah pengulangan dengan pusat awal berbeda; hasil terbaik dipakai"),
                ("random_state", p["random_state"],
                 "Seed acak agar hasil dapat direproduksi"),
                ("init", p["init"],
                 "Metode pemilihan pusat awal (nilai bawaan scikit-learn)"),
                ("max_iter", p["max_iter"],
                 "Batas iterasi maksimum (nilai bawaan scikit-learn)"),
                ("tol", p["tol"],
                 "Ambang perubahan pusat cluster untuk dianggap konvergen (nilai bawaan scikit-learn)"),
                ("algorithm", p.get("algorithm", "-"),
                 "Varian algoritma K-Means (nilai bawaan scikit-learn)"),
            ])

            st.markdown("#### Data dan prapemrosesan")
            tabel_parameter([
                ("Fitur",
                 f"IPM laki-laki dan IPM perempuan {tahun_cluster}",
                 "Dua variabel yang dikelompokkan"),
                ("Normalisasi",
                 f"MinMaxScaler, rentang {scaler_ipm.feature_range}",
                 "Setiap kolom IPM diubah ke skala 0–1 sebelum clustering"),
                ("Kandidat k",
                 f"{min(k_uji)} sampai {max(k_uji)}",
                 "Setiap nilai k diuji, lalu dipilih yang terbaik"),
                ("Kriteria memilih k", "Silhouette score tertinggi",
                 "Semakin tinggi, pemisahan antar-cluster relatif semakin baik"),
                ("Penamaan kategori", "Urut rata-rata IPM gabungan",
                 "Cluster diberi label relatif rendah sampai tinggi"),
                ("Pustaka", f"scikit-learn {sklearn.__version__}", ""),
            ])

        with tab_hasil:
            profil = hasil_cluster.groupby("Karakteristik").agg(
                Jumlah_Wilayah=("Nama_Wilayah", "count"),
                Rata_IPM_L=("IPM_L", "mean"),
                Rata_IPM_P=("IPM_P", "mean"),
                Rata_Selisih=("Selisih_IPM", "mean")
            ).reset_index()

            # Urutkan kategori dari rata-rata IPM terendah ke tertinggi
            urutan_nama = (
                hasil_cluster.groupby("Karakteristik")
                .apply(
                    lambda x: (
                        x["IPM_L"].mean()
                        + x["IPM_P"].mean()
                    ) / 2,
                    include_groups=False
                )
                .sort_values()
                .index.tolist()
            )

            profil["Karakteristik"] = pd.Categorical(
                profil["Karakteristik"],
                categories=urutan_nama,
                ordered=True
            )

            profil = profil.sort_values("Karakteristik")

            st.dataframe(
                profil[
                    [
                        "Karakteristik",
                        "Jumlah_Wilayah",
                        "Rata_IPM_L",
                        "Rata_IPM_P",
                        "Rata_Selisih"
                    ]
                ].rename(columns={
                    "Karakteristik": "Kategori IPM",
                    "Jumlah_Wilayah": "Jumlah wilayah",
                    "Rata_IPM_L": "Rata-rata IPM laki-laki",
                    "Rata_IPM_P": "Rata-rata IPM perempuan",
                    "Rata_Selisih": "Rata-rata selisih IPM"
                }).round(2),
                use_container_width=True,
                hide_index=True
            )

            # Visualisasi hasil clustering
            st.subheader("Visualisasi Karakteristik Wilayah")

            fig, ax = plt.subplots(figsize=(9, 5))

            for kategori in urutan_nama:
                bagian = hasil_cluster[
                    hasil_cluster["Karakteristik"] == kategori
                ]

                ax.scatter(
                    bagian["IPM_L"],
                    bagian["IPM_P"],
                    label=kategori,
                    alpha=0.75
                )

            ax.set_xlabel(f"IPM laki-laki tahun {tahun_cluster}")
            ax.set_ylabel(f"IPM perempuan tahun {tahun_cluster}")
            ax.set_title(f"Kemiripan Karakteristik IPM Antarwilayah ({tahun_cluster})"
            )
            ax.legend(title="Kategori IPM")
            ax.spines[["top", "right"]].set_visible(False)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            profil_urut = profil.sort_values(
                "Rata_IPM_L",
                ascending=False
            )

            kategori_atas = profil_urut.iloc[0]
            kategori_bawah = profil_urut.iloc[-1]

            kartu_temuan(
                "Perbedaan karakteristik wilayah",
                f"{kategori_atas['Karakteristik']} memiliki rata-rata "
                f"IPM laki-laki sebesar "
                f"{format_angka(kategori_atas['Rata_IPM_L'])}, "
                f"sedangkan {kategori_bawah['Karakteristik']} memiliki "
                f"rata-rata sebesar "
                f"{format_angka(kategori_bawah['Rata_IPM_L'])}. "
                f"Kategori dibentuk berdasarkan kemiripan nilai IPM "
                f"laki-laki dan perempuan tahun {tahun_cluster}."
            )



        with tab_eval:
            tabel_k = pd.DataFrame({
                "Jumlah cluster (k)": [s[0] for s in skor_cluster],
                "Silhouette score": [s[1] for s in skor_cluster],
                "Inertia": [s[2].inertia_ for s in skor_cluster],
            })
            tabel_k["Terpilih"] = np.where(
                tabel_k["Jumlah cluster (k)"] == jumlah_cluster, "✔", ""
            )

            fig, ax = plt.subplots(figsize=(8, 3.8))
            warna = [
                "#34765D" if k == jumlah_cluster else "#B8CFC4"
                for k in tabel_k["Jumlah cluster (k)"]
            ]
            ax.bar(
                tabel_k["Jumlah cluster (k)"].astype(str),
                tabel_k["Silhouette score"],
                color=warna
            )
            ax.set_xlabel("Jumlah cluster (k)")
            ax.set_ylabel("Silhouette score")
            ax.set_title(
                "Silhouette score untuk setiap k",
                loc="left", fontsize=13, fontweight="bold"
            )
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(
                tabel_k.round(4),
                use_container_width=True,
                hide_index=True
            )

            kartu_temuan(
                "Cara membaca evaluasi",
                f"Nilai k = {jumlah_cluster} dipilih karena memberi "
                f"silhouette score tertinggi ({silhouette_terbaik:.4f}). "
                "Inertia adalah jumlah jarak kuadrat titik ke pusat "
                "clusternya dan selalu turun saat k bertambah, sehingga "
                "tidak dipakai sebagai kriteria pemilihan k."
            )

        with tab_daftar:
            if hasil_cluster is not None:
                cluster_tampil = hasil_cluster.copy()

                if provinsi_pilihan != "Semua Provinsi":
                    cluster_tampil = cluster_tampil[
                        cluster_tampil["Provinsi"] == provinsi_pilihan
                    ]

                if cluster_tampil.empty:
                    st.info("Tidak ada data clustering untuk provinsi ini.")
                else:
                    urutan_ipm = (
                        cluster_tampil.groupby("Karakteristik")
                        .apply(
                            lambda x: (
                                x["IPM_L"].mean()
                                + x["IPM_P"].mean()
                            ) / 2,
                            include_groups=False
                        )
                        .sort_values()
                        .index.tolist()
                    )

                    for kategori in urutan_ipm:
                        bagian = cluster_tampil[
                            cluster_tampil["Karakteristik"] == kategori
                        ].copy()

                        rata_l = bagian["IPM_L"].mean()
                        rata_p = bagian["IPM_P"].mean()

                        with st.expander(
                            f"{kategori} — {len(bagian)} wilayah"
                        ):
                            c1, c2 = st.columns(2)

                            c1.metric(
                                "Rata-rata IPM laki-laki",
                                format_angka(rata_l)
                            )

                            c2.metric(
                                "Rata-rata IPM perempuan",
                                format_angka(rata_p)
                            )

                            daftar = bagian[
                                [
                                    "Provinsi",
                                    "Kabupaten/Kota",
                                    "IPM_L",
                                    "IPM_P"
                                ]
                            ].sort_values(
                                "IPM_L",
                                ascending=True
                            )

                            daftar.columns = [
                                "Provinsi",
                                "Kabupaten/Kota",
                                f"IPM laki-laki {tahun_cluster}",
                                f"IPM perempuan {tahun_cluster}"
                            ]

                            st.dataframe(
                                daftar.round(2),
                                use_container_width=True,
                                hide_index=True
                            )

                            st.download_button(
                                f"Unduh daftar {kategori}",
                                data=daftar.to_csv(
                                    index=False
                                ).encode("utf-8-sig"),
                                file_name=(
                                    kategori.lower()
                                    .replace(" ", "_")
                                    + ".csv"
                                ),
                                mime="text/csv",
                                key=f"unduh_ipm_{kategori}"
                            )

            else:
                st.info(
                    "Pengelompokan IPM belum tersedia. "
                    "Periksa jumlah data dan hasil clustering."
                )



elif halaman == "Regresi Linear":
    header_model(
        "Regresi Linear",
        "Memodelkan hubungan Rata-rata Lama Sekolah (RLS) dengan IPM, "
        "lalu mengukur kinerjanya pada data uji."
    )

    if not hasil_regresi:
        st.info(
            "Model regresi belum tersedia karena data valid "
            "belum mencukupi."
        )
    else:
        st.caption(
            "Model dilatih pada seluruh wilayah dan seluruh tahun "
            "2010–2013, sehingga tidak berubah mengikuti pilihan tahun "
            "atau provinsi di sidebar."
        )

        tab_konfig, tab_hasil, tab_simulasi = st.tabs([
            "⚙️ Konfigurasi Model",
            "📊 Hasil dan Evaluasi",
            "🔮 Simulasi Prediksi"
        ])

        contoh = next(iter(hasil_regresi.values()))

        with tab_konfig:
            p = contoh["model"].get_params()
            sp = contoh["params_split"]
            porsi_uji = int(round(sp["test_size"] * 100))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Algoritma", "Linear Regression")
            c2.metric("Variabel input (X)", "RLS")
            c3.metric("Target (y)", "IPM L dan P")
            c4.metric("Latih : uji", f"{100 - porsi_uji} : {porsi_uji}")

            st.markdown("#### Parameter model")
            tabel_parameter([
                ("fit_intercept", p["fit_intercept"],
                 "Model menghitung konstanta (intercept) (nilai bawaan scikit-learn)"),
                ("positive", p.get("positive", False),
                 "Koefisien tidak dipaksa positif (nilai bawaan scikit-learn)"),
                ("copy_X", p["copy_X"],
                 "Data input disalin sebelum dilatih (nilai bawaan scikit-learn)"),
                ("n_jobs", p["n_jobs"],
                 "Jumlah proses paralel (nilai bawaan scikit-learn)"),
            ])

            st.markdown("#### Pembagian data latih dan uji")
            tabel_parameter([
                ("Metode", "GroupShuffleSplit",
                 "Pembagian acak berdasarkan kelompok wilayah"),
                ("Kelompok", "Nama wilayah",
                 "Wilayah yang sama tidak muncul di data latih dan uji sekaligus"),
                ("n_splits", sp["n_splits"], "Pembagian dilakukan satu kali"),
                ("test_size", sp["test_size"], "Porsi data uji"),
                ("random_state", sp["random_state"],
                 "Seed acak agar pembagian dapat direproduksi"),
                ("Data", "RLS dan IPM tahun 2010–2013 digabung",
                 "Setiap pasangan wilayah-tahun menjadi satu baris data"),
                ("Normalisasi", "Tidak digunakan",
                 "Model memakai nilai asli: RLS dalam tahun, IPM dalam poin"),
                ("Pustaka", f"scikit-learn {sklearn.__version__}", ""),
            ])

            st.markdown("#### Hasil estimasi model")
            baris_hasil = []
            for nama, hasil in hasil_regresi.items():
                b0 = float(hasil["model"].intercept_)
                b1 = float(hasil["model"].coef_[0])
                baris_hasil.append({
                    "Target": f"IPM {nama.lower()}",
                    "Intercept": round(b0, 4),
                    "Koefisien RLS": round(b1, 4),
                    "Persamaan": f"IPM = {b0:.2f} + {b1:.2f} × RLS",
                    "Data total": hasil["n_total"],
                    "Data latih": len(hasil["data_latih"]),
                    "Data uji": len(hasil["data_uji"]),
                    "Jumlah wilayah": hasil["n_wilayah"],
                })
            st.dataframe(
                pd.DataFrame(baris_hasil),
                use_container_width=True,
                hide_index=True
            )

        with tab_hasil:
            for nama, hasil in hasil_regresi.items():
                st.markdown(f"#### IPM {nama.lower()}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MAE", format_angka(hasil["mae"]))
                m2.metric("RMSE", format_angka(hasil["rmse"]))
                m3.metric("R² data uji", format_angka(hasil["r2"]))
                m4.metric("Jumlah data uji", len(hasil["data_uji"]))

                latih = hasil["data_latih"]
                uji = hasil["data_uji"]
                kiri, kanan = st.columns(2)

                with kiri:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(latih["RLS"], latih["IPM"], alpha=0.3,
                               color="#9DB8AB", label="Data latih")
                    ax.scatter(uji["RLS"], uji["IPM"], alpha=0.85,
                               color="#183D32", label="Data uji")
                    xs = np.linspace(
                        min(latih["RLS"].min(), uji["RLS"].min()),
                        max(latih["RLS"].max(), uji["RLS"].max()),
                        100
                    )
                    ax.plot(
                        xs,
                        hasil["model"].predict(pd.DataFrame({"RLS": xs})),
                        color="#D9822B", linewidth=2, label="Garis regresi"
                    )
                    ax.set_xlabel("Rata-rata lama sekolah (tahun)")
                    ax.set_ylabel(f"IPM {nama.lower()}")
                    ax.set_title("Garis regresi", loc="left", fontweight="bold")
                    ax.legend()
                    ax.spines[["top", "right"]].set_visible(False)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with kanan:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(hasil["aktual"], hasil["prediksi"],
                               alpha=0.8, color="#34765D")
                    batas = [
                        min(hasil["aktual"].min(), hasil["prediksi"].min()),
                        max(hasil["aktual"].max(), hasil["prediksi"].max()),
                    ]
                    ax.plot(batas, batas, linestyle="--", color="#D9822B",
                            label="Prediksi sempurna")
                    ax.set_xlabel("IPM aktual")
                    ax.set_ylabel("IPM prediksi")
                    ax.set_title("Aktual vs prediksi (data uji)",
                                 loc="left", fontweight="bold")
                    ax.legend()
                    ax.spines[["top", "right"]].set_visible(False)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                if hasil["r2"] >= 0.7:
                    keterangan = (
                        "Pada data uji, hubungan yang dipelajari model memiliki "
                        "kemampuan menjelaskan variasi IPM yang relatif tinggi."
                    )
                elif hasil["r2"] >= 0.3:
                    keterangan = (
                        "Pada data uji, kemampuan model menjelaskan variasi IPM "
                        "berada pada tingkat sedang."
                    )
                else:
                    keterangan = (
                        "Pada data uji, kemampuan model menjelaskan variasi IPM "
                        "masih terbatas. Nilai ini perlu ditafsirkan bersama "
                        "ukuran galat dan jumlah data uji."
                    )

                kartu_temuan(
                    f"Evaluasi hubungan RLS dengan IPM {nama.lower()}",
                    f"{keterangan} Nilai R² adalah {format_angka(hasil['r2'])}, "
                    f"dengan galat absolut rata-rata {format_angka(hasil['mae'])} "
                    "poin IPM. Hasil ini menunjukkan kinerja pada data uji, "
                    "bukan bukti bahwa pendidikan sendiri menyebabkan "
                    "perubahan IPM."
                )

        with tab_simulasi:
            st.write(
                "Geser nilai RLS untuk melihat IPM yang diprediksi model. "
                "Rentang slider dibatasi pada rentang RLS yang ada di data."
            )

            semua_rls = pd.concat(
                [h["data_latih"]["RLS"] for h in hasil_regresi.values()]
                + [h["data_uji"]["RLS"] for h in hasil_regresi.values()]
            )
            rls_min = float(np.floor(semua_rls.min() * 10) / 10)
            rls_max = float(np.ceil(semua_rls.max() * 10) / 10)

            nilai_rls = st.slider(
                "Rata-rata lama sekolah (tahun)",
                min_value=rls_min,
                max_value=rls_max,
                value=float(round(semua_rls.median(), 1)),
                step=0.1,
                key="slider_simulasi_rls"
            )

            kolom_sim = st.columns(len(hasil_regresi))
            for wadah, (nama, hasil) in zip(kolom_sim, hasil_regresi.items()):
                prediksi_sim = hasil["model"].predict(
                    pd.DataFrame({"RLS": [nilai_rls]})
                )[0]
                wadah.metric(
                    f"Prediksi IPM {nama.lower()}",
                    format_angka(prediksi_sim)
                )

            st.caption(
                "Ini estimasi statistik dari pola data 2010–2013, bukan "
                "proyeksi kebijakan dan bukan bukti hubungan sebab-akibat."
            )


# =========================================================
# 12. HALAMAN TENTANG DATA
# =========================================================


elif halaman == "Tentang Data":
    st.title("Tentang Data")
    st.write(
        "Halaman ini menjelaskan sumber data, tahapan prapemrosesan, "
        "normalisasi, serta metode analisis yang digunakan."
    )

    # 1. Sumber data
    st.header("1. Sumber Data")

    st.markdown("""
    Penelitian ini menggunakan dua dataset yang bersumber dari Badan
    Pusat Statistik (BPS), yaitu:

    **Dataset Indeks Pembangunan Manusia (IPM)**

    Memuat nilai IPM menurut jenis kelamin, yaitu laki-laki dan
    perempuan, pada tahun 2010–2013.

    [Sumber data IPM menurut jenis kelamin – BPS](https://www.bps.go.id/id/statistics-table/2/NDYyIzI=/indeks-pembangunan-manusia--ipm--menurut-jenis-kelamin.html)

    **Dataset Rata-rata Lama Sekolah (RLS)**

    Memuat indikator rata-rata lama sekolah pada tahun 2010–2013.

    [Sumber data RLS – BPS](https://www.bps.go.id/id/statistics-table/2/NSMy/rata-rata-lama-sekolah-tahun.html)

    Kedua dataset disatukan berdasarkan identitas wilayah kabupaten/kota
    dan provinsi agar indikator IPM dan RLS dapat dianalisis bersama.
    """)

    st.info(
        f"Data memuat {len(df)} baris wilayah dan "
        f"{len(KOLOM_NUMERIK)} kolom indikator numerik, "
        "yang terdiri dari 8 kolom IPM dan 4 kolom RLS."
    )

    # 2. Prapemrosesan data
    st.header("2. Prapemrosesan Data")

    st.markdown("""
    Prapemrosesan dilakukan untuk mempersiapkan data sebelum dianalisis.
    Tahapan yang dilakukan meliputi:

    **a. Pemeriksaan duplikasi**

    Pemeriksaan data duplikat dilakukan pada tahap pengolahan awal
    menggunakan Microsoft Excel untuk memeriksa kemungkinan adanya
    pencatatan wilayah yang berulang.

    **b. Pemeriksaan outlier**

    Pemeriksaan outlier juga dilakukan menggunakan Microsoft Excel
    untuk meninjau nilai yang menyimpang dari pola data. Pemeriksaan
    ini dilakukan sebelum data digunakan dalam analisis.

    **c. Penanganan data kosong**

    Data kosong ditangani berdasarkan posisi dan ketersediaan data.
    Interpolasi linear digunakan untuk memperkirakan nilai kosong
    di antara data yang tersedia, sedangkan ekstrapolasi linear
    digunakan untuk memperkirakan nilai di luar rentang data yang
    tersedia dengan memanfaatkan kecenderungan data.

    Baris wilayah yang belum terbentuk pada tahun tertentu dan belum
    memiliki data historis yang relevan dihapus dari dataset karena
    nilai pada periode tersebut memang belum tersedia.

    **d. Konversi tipe data**

    Setelah file Excel dibaca oleh Python, kolom indikator dikonversi
    menjadi tipe numerik menggunakan `pd.to_numeric()` dengan
    `errors="coerce"`. Nilai yang tidak dapat dikonversi menjadi
    angka akan ditandai sebagai `NaN` atau nilai kosong.
    Proses ini tidak mengisi nilai kosong secara otomatis.
    """)

    # Ringkasan kelengkapan data
    st.subheader("Kelengkapan Data Saat Dibaca Python")

    ringkasan_kosong = pd.DataFrame({
        "Indikator": KOLOM_NUMERIK,
        "Jumlah Data Kosong": [
            int(df[kolom].isna().sum()) for kolom in KOLOM_NUMERIK
        ]
    })

    st.dataframe(ringkasan_kosong, use_container_width=True, hide_index=True)

    st.caption(
        "Ringkasan ini menunjukkan nilai kosong yang masih tersisa "
        "di dalam dataframe setelah file Excel dibaca dan dikonversi "
        "menjadi numerik."
    )

    # 3. Normalisasi
    st.header("3. Normalisasi dengan Min-Max Scaling")

    st.markdown("""
    Normalisasi dilakukan menggunakan `MinMaxScaler` dari pustaka
    scikit-learn untuk mengubah skala nilai indikator menjadi rentang
    0 sampai 1.

    Normalisasi IPM dilakukan terhadap seluruh kolom IPM, sedangkan
    normalisasi RLS dilakukan terhadap seluruh kolom RLS. Pemisahan
    ini mempertahankan pengelompokan indikator sesuai jenisnya.

    Normalisasi terutama digunakan dalam analisis K-Means agar
    perbedaan skala nilai tidak mendominasi proses pengelompokan.
    """)

    st.latex(r"X' = \frac{X-X_{\min}}{X_{\max}-X_{\min}}")

    # 4. K-Means
    st.header("4. Metode K-Means Clustering")

    st.markdown("""
    K-Means merupakan metode pengelompokan data berdasarkan kemiripan
    karakteristik. Dalam penelitian ini, K-Means digunakan untuk
    mengelompokkan kabupaten/kota berdasarkan nilai IPM laki-laki
    dan perempuan pada tahun yang dipilih.

    Tahapan analisis meliputi:
    1. Memilih tahun analisis.
    2. Mengambil nilai IPM laki-laki dan perempuan pada tahun tersebut.
    3. Menggunakan data yang memiliki kedua nilai IPM.
    4. Menguji beberapa jumlah cluster dan menghitung
       *silhouette score* untuk menilai kualitas pengelompokan.
    5. Memilih jumlah cluster dengan nilai silhouette score terbaik.
    6. Mengelompokkan wilayah dan mengurutkan label cluster
       berdasarkan rata-rata IPM.

    Nilai silhouette score yang lebih tinggi menunjukkan pemisahan
    cluster yang relatif lebih baik. Hasilnya tetap perlu
    diinterpretasikan berdasarkan karakteristik data.
    """)
    st.markdown("#### Hasil Evaluasi K-Means")

    st.write(
        f"Jumlah cluster terbaik: {jumlah_cluster}"
    )

    st.metric(
        "Silhouette Score",
        f"{silhouette_terbaik:.4f}"
        if silhouette_terbaik is not None
        else "Tidak tersedia"
    )

    # 5. Regresi linear
    st.header("5. Metode Regresi Linear")

    st.markdown("""
    Regresi linear digunakan untuk menganalisis hubungan Rata-rata
    Lama Sekolah (RLS) dengan Indeks Pembangunan Manusia (IPM), serta
    memprediksi nilai IPM berdasarkan nilai RLS.

    Analisis dilakukan secara terpisah untuk IPM laki-laki dan
    perempuan menggunakan data tahun 2010–2013. Data yang tidak
    memiliki nilai RLS atau IPM yang diperlukan tidak disertakan
    dalam pemodelan terkait.

    Data dibagi menjadi data latih dan data uji menggunakan
    `GroupShuffleSplit`, dengan kabupaten/kota sebagai kelompok.
    Pembagian berdasarkan kelompok membantu mencegah wilayah yang
    sama muncul sekaligus dalam data latih dan data uji.

    Kinerja model dievaluasi menggunakan tiga metrik:
    - **MAE (Mean Absolute Error):** rata-rata selisih absolut antara
      nilai aktual dan prediksi. Nilai lebih rendah menunjukkan
      kesalahan prediksi yang lebih kecil.
    - **RMSE (Root Mean Squared Error):** mengukur besarnya kesalahan
      prediksi dengan memberikan penalti lebih besar pada kesalahan
      yang besar. Nilai lebih rendah lebih baik.
    - **R² (Koefisien Determinasi):** menunjukkan seberapa baik model
      menjelaskan variasi nilai target. Nilai yang mendekati 1
      umumnya menunjukkan kemampuan penjelasan yang lebih baik.
    """)

    # 6. Batasan interpretasi
    st.header("6. Catatan Pengolahan Data")

    st.markdown("""
    Hasil analisis dipengaruhi oleh kelengkapan data, estimasi nilai
    melalui interpolasi atau ekstrapolasi, serta cakupan wilayah dan
    tahun yang tersedia. Nilai hasil estimasi bukan pengukuran asli
    BPS, sehingga perlu dipertimbangkan saat menafsirkan hasil
    pengelompokan dan prediksi.

    Regresi linear menunjukkan hubungan statistik antara RLS dan IPM,
    tetapi hubungan tersebut tidak dengan sendirinya membuktikan
    hubungan sebab-akibat.
    """)

# =========================================================
# 13. FOOTER
# =========================================================

st.divider()
st.caption(
    "Dashboard Statistik Pembangunan Manusia dan Pendidikan · "
    "Sumber data bersumber Badan Pusat Statistik."
)