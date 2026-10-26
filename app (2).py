from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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


# Lokasi file Excel dalam folder yang sama dengan app.py
FILE_DATA = Path(__file__).resolve().parent / "Data_Rata_Rata_Rapi.xlsx"


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
    /* Latar utama */
    .stApp {
        background-color: #F5F7F5;
        color: #24352D;
    }

    /* Warna teks utama */
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp .stMarkdown,
    .stApp [data-testid="stCaptionContainer"] {
        color: #24352D;
    }

    /* Judul halaman */
    h1, h2, h3, h4 {
        color: #183D32 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #183D32;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #FFFFFF !important;
    }

    /* Pilihan navigasi sidebar */
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        color: #FFFFFF !important;
    }

    /* Area konten */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* Header utama */
    .hero {
        background: linear-gradient(120deg, #183D32, #34765D);
        padding: 28px;
        border-radius: 18px;
        color: #FFFFFF;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        margin-bottom: 8px;
    }

    .hero p {
        color: #E4F0E8 !important;
        margin-bottom: 0;
    }

    /* Kartu temuan statistik */
    .insight {
        background-color: #FFFFFF;
        border: 1px solid #E1E8E3;
        border-left: 5px solid #34765D;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0 16px 0;
        color: #24352D !important;
    }

    .insight-title {
        font-weight: 700;
        color: #183D32 !important;
        margin-bottom: 6px;
    }

    .insight div {
        color: #24352D !important;
    }

    /* Kartu angka statistik */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E1E8E3;
        padding: 16px;
        border-radius: 14px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #52645A !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #183D32 !important;
    }

    /* Tabel */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
    }

    /* Kotak input dan pilihan */
    .stSelectbox div[data-baseweb="select"] {
        color: #24352D;
    }

    /* Tombol */
    .stDownloadButton button {
        background-color: #183D32;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
    }

    .stDownloadButton button:hover {
        background-color: #34765D;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

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

# Simpan kondisi data sebelum proses cleaning untuk dokumentasi.
jumlah_baris_awal = len(df)
jumlah_duplikat_awal = int(df.duplicated().sum())
jumlah_kosong_awal = int(df[KOLOM_NUMERIK].isna().sum().sum())

# Hapus baris duplikat yang identik sepenuhnya.
df = df.drop_duplicates().copy()

# Isi nilai kosong antar-tahun dengan interpolasi atau ekstrapolasi linier.
# Dilakukan per wilayah dan per indikator (misalnya IPM_L dan RLS terpisah).
def isi_linier_per_wilayah(baris, kolom_tahun):
    nilai = pd.to_numeric(baris[kolom_tahun], errors="coerce").to_numpy(dtype=float)
    valid = np.flatnonzero(np.isfinite(nilai))
    if len(valid) < 2:
        # Jika kurang dari dua titik diketahui, tidak cukup dasar untuk
        # memperkirakan tren linier; nilai kosong dibiarkan tetap kosong.
        return pd.Series(nilai, index=kolom_tahun)

    kosong = np.flatnonzero(~np.isfinite(nilai))
    for idx in kosong:
        if idx < valid[0]:
            i1, i2 = valid[0], valid[1]
        elif idx > valid[-1]:
            i1, i2 = valid[-2], valid[-1]
        else:
            sebelum = valid[valid < idx]
            sesudah = valid[valid > idx]
            i1, i2 = sebelum[-1], sesudah[0]

        kemiringan = (nilai[i2] - nilai[i1]) / (i2 - i1)
        nilai[idx] = nilai[i1] + kemiringan * (idx - i1)

    return pd.Series(nilai, index=kolom_tahun)

jumlah_diperkirakan = 0
for awalan in ["IPM_L", "IPM_P", "RLS"]:
    kolom_tahun = [f"{awalan}_{tahun}" for tahun in TAHUN]
    sebelum = int(df[kolom_tahun].isna().sum().sum())
    df[kolom_tahun] = df.apply(
        lambda baris: isi_linier_per_wilayah(baris, kolom_tahun), axis=1
    )
    jumlah_diperkirakan += sebelum - int(df[kolom_tahun].isna().sum().sum())

jumlah_baris_bersih = len(df)
jumlah_kosong_setelah = int(df[KOLOM_NUMERIK].isna().sum().sum())
jumlah_duplikat_setelah = int(df.duplicated().sum())

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


def grafik_batang(data, kolom, judul, label_y, jumlah=10):
    valid = data.dropna(subset=[kolom]).nlargest(jumlah, kolom)

    if valid.empty:
        st.info("Belum ada data yang dapat ditampilkan.")
        return

    valid = valid.sort_values(kolom)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(valid["Nama_Wilayah"], valid[kolom])
    ax.set_title(judul, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel(label_y)
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
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

scaler = MinMaxScaler()
data_normalisasi = df.copy()

data_normalisasi[KOLOM_NUMERIK] = scaler.fit_transform(
    df[KOLOM_NUMERIK]
)

# Pengelompokan karakteristik wilayah berdasarkan IPM tahun 2013
kolom_cluster = ["IPM_L_2013", "IPM_P_2013"]

data_cluster = data_normalisasi.dropna(
    subset=kolom_cluster
).copy()

hasil_cluster = None
jumlah_cluster = None
silhouette_terbaik = None

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
        jumlah_cluster, silhouette_terbaik, model_terbaik, label_terbaik = (
            max(skor_cluster, key=lambda x: x[1])
        )

        hasil_cluster = df.loc[
            data_cluster.index,
            [
                "Provinsi", "Kabupaten/Kota", "Nama_Wilayah",
                "IPM_L_2013", "IPM_P_2013"
            ]
        ].copy()

        hasil_cluster["Kelompok"] = label_terbaik + 1

        hasil_cluster["Selisih_IPM"] = (
            hasil_cluster["IPM_L_2013"]
            - hasil_cluster["IPM_P_2013"]
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
            "data_uji": data_model.iloc[indeks_uji].copy()
        }


# =========================================================
# 5. SIDEBAR DAN FILTER
# =========================================================

st.sidebar.markdown("## 📊 Statistik Wilayah")
st.sidebar.caption("Pembangunan Manusia & Pendidikan")

halaman = st.sidebar.radio(
    "Navigasi",
    [
        "Ringkasan",
        "Statistik IPM",
        "Statistik Pendidikan",
        "Perbandingan Wilayah",
        "Hasil Analisis Model",
        "Persiapan Data (5–7)",
        "Tentang Data"
    ]
)

tahun_pilihan = st.sidebar.selectbox(
    "Pilih tahun",
    TAHUN,
    index=len(TAHUN) - 1
)

provinsi_pilihan = st.sidebar.selectbox(
    "Pilih provinsi",
    ["Semua Provinsi"] + sorted(df["Provinsi"].unique().tolist())
)

if provinsi_pilihan == "Semua Provinsi":
    data_tampilan = df.copy()
else:
    data_tampilan = df[
        df["Provinsi"] == provinsi_pilihan
    ].copy()

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
    f"Periode data: 2010–2013 · Tahun yang dipilih: {tahun_pilihan} "
    f"· Cakupan: {provinsi_pilihan}"
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

    


# =========================================================
# 8. HALAMAN HASIL ANALISIS MODEL
# =========================================================

elif halaman == "Hasil Analisis Model":
    st.subheader("Hasil Analisis Model")
    st.write(
        "Bagian ini memisahkan hasil pemodelan dari statistik deskriptif. "
        "Model digunakan untuk mengelompokkan wilayah berdasarkan kemiripan IPM "
        "dan menilai hubungan statistik antara RLS dengan IPM."
    )

    tab_cluster, tab_regresi = st.tabs([
        "Pengelompokan Wilayah",
        "Hubungan RLS dan IPM"
    ])

    with tab_cluster:
        st.markdown("### Algoritma: K-Means Clustering")
        st.write(
            "K-Means mengelompokkan wilayah dengan nilai IPM laki-laki dan "
            "perempuan yang memiliki pola serupa pada tahun 2013. Nomor kelompok "
            "hanya label hasil komputasi, bukan peringkat resmi pembangunan."
        )

        with st.expander("Lihat konfigurasi dan parameter K-Means", expanded=True):
            st.markdown("""
            - **Fitur yang digunakan:** IPM laki-laki 2013 (`IPM_L_2013`) dan IPM perempuan 2013 (`IPM_P_2013`).
            - **Pra-pemrosesan:** `MinMaxScaler` untuk menormalisasi kolom numerik ke rentang 0–1; K-Means menggunakan dua fitur IPM tahun 2013 yang sudah dinormalisasi.
            - **Jumlah kelompok (`n_clusters`):** dipilih otomatis dari kandidat 2 sampai maksimum 6 (atau jumlah data valid dikurangi 1 jika lebih kecil).
            - **Kriteria pemilihan jumlah kelompok:** nilai *Silhouette Score* tertinggi di antara kandidat yang dapat dihitung.
            - **`random_state`:** `42` agar hasil dapat direproduksi.
            - **`n_init`:** `10` percobaan inisialisasi pusat kelompok untuk setiap kandidat jumlah kelompok.
            - **Evaluasi pengelompokan:** *Silhouette Score*; semakin mendekati 1 umumnya menunjukkan pemisahan kelompok yang lebih baik.
            """)

        if hasil_cluster is not None:
            c1, c2 = st.columns(2)
            c1.metric("Jumlah kelompok terpilih", str(jumlah_cluster))
            c2.metric("Silhouette Score", format_angka(silhouette_terbaik, 3))

            profil = hasil_cluster.groupby("Kelompok").agg(
                Jumlah_Wilayah=("Nama_Wilayah", "count"),
                Rata_IPM_L=("IPM_L_2013", "mean"),
                Rata_IPM_P=("IPM_P_2013", "mean"),
                Rata_Selisih=("Selisih_IPM", "mean")
            ).reset_index()

            profil["Karakteristik"] = "Kelompok " + profil["Kelompok"].astype(str)
            profil_tampil = profil[[
                "Karakteristik", "Jumlah_Wilayah", "Rata_IPM_L",
                "Rata_IPM_P", "Rata_Selisih"
            ]].rename(columns={
                "Jumlah_Wilayah": "Jumlah wilayah",
                "Rata_IPM_L": "Rata-rata IPM laki-laki 2013",
                "Rata_IPM_P": "Rata-rata IPM perempuan 2013",
                "Rata_Selisih": "Rata-rata selisih IPM"
            })
            st.dataframe(profil_tampil.round(2), use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(9, 5))
            for kelompok in sorted(hasil_cluster["Kelompok"].unique()):
                bagian_kelompok = hasil_cluster[hasil_cluster["Kelompok"] == kelompok]
                ax.scatter(
                    bagian_kelompok["IPM_L_2013"],
                    bagian_kelompok["IPM_P_2013"],
                    label=f"Kelompok {kelompok}",
                    alpha=0.7
                )
            ax.set_xlabel("IPM laki-laki 2013")
            ax.set_ylabel("IPM perempuan 2013")
            ax.set_title("Sebaran Wilayah berdasarkan Kelompok IPM 2013")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("#### Data wilayah hasil pengelompokan")
            st.dataframe(
                hasil_cluster[[
                    "Provinsi", "Kabupaten/Kota", "IPM_L_2013",
                    "IPM_P_2013", "Selisih_IPM", "Kelompok"
                ]].sort_values("Kelompok"),
                use_container_width=True,
                hide_index=True
            )

            profil_urut = profil.sort_values("Rata_IPM_L", ascending=False)
            kelompok_atas = profil_urut.iloc[0]
            kelompok_bawah = profil_urut.iloc[-1]
            kartu_temuan(
                "Cara membaca hasil kelompok",
                f"Kelompok {int(kelompok_atas['Kelompok'])} memiliki rata-rata IPM laki-laki tertinggi "
                f"({format_angka(kelompok_atas['Rata_IPM_L'])}), sedangkan Kelompok "
                f"{int(kelompok_bawah['Kelompok'])} memiliki rata-rata terendah "
                f"({format_angka(kelompok_bawah['Rata_IPM_L'])}). Ini merupakan ringkasan pola "
                "kelompok berdasarkan IPM 2013, bukan bukti sebab-akibat atau peringkat resmi."
            )
        else:
            st.info("Pengelompokan belum dapat ditampilkan karena data valid belum mencukupi.")

    with tab_regresi:
        st.markdown("### Algoritma: Regresi Linear")
        st.write(
            "Regresi linear digunakan untuk mempelajari hubungan antara rata-rata lama sekolah "
            "(RLS) sebagai variabel masukan dan IPM sebagai variabel target. Analisis dilakukan "
            "secara terpisah untuk IPM laki-laki dan IPM perempuan, dengan data tahun 2010–2013."
        )

        with st.expander("Lihat konfigurasi dan parameter regresi linear", expanded=True):
            st.markdown("""
            - **Algoritma:** `LinearRegression` dari scikit-learn.
            - **Variabel masukan (`X`):** RLS (`RLS`) satu fitur.
            - **Variabel target (`y`):** IPM (`IPM`), dianalisis terpisah untuk laki-laki dan perempuan.
            - **Rentang data:** data RLS dan IPM yang tersedia dari tahun 2010–2013 digabung untuk masing-masing jenis kelamin.
            - **Pembagian data:** `GroupShuffleSplit`, 80% data kelompok wilayah untuk pelatihan dan 20% untuk pengujian (`test_size=0.2`).
            - **Pengelompokan saat membagi data:** `Nama_Wilayah`, sehingga satu wilayah tidak dibagi sekaligus ke data latih dan data uji.
            - **`n_splits`:** `1` pembagian.
            - **`random_state`:** `42`.
            - **Pengaturan regresi:** `fit_intercept=True` (nilai bawaan), sehingga model menghitung intersep.
            - **Metrik evaluasi:** MAE, RMSE, dan R² dihitung pada data uji.
            """)

        if hasil_regresi:
            pilihan_regresi = st.selectbox(
                "Pilih hasil regresi",
                list(hasil_regresi.keys()),
                key="pilihan_hasil_regresi"
            )
            hasil = hasil_regresi[pilihan_regresi]

            m1, m2, m3 = st.columns(3)
            m1.metric("MAE (poin IPM)", format_angka(hasil["mae"]))
            m2.metric("RMSE (poin IPM)", format_angka(hasil["rmse"]))
            m3.metric("R² data uji", format_angka(hasil["r2"]))

            st.caption(
                "MAE dan RMSE menunjukkan besar galat prediksi (lebih kecil umumnya lebih baik). "
                "R² menunjukkan seberapa jauh variasi IPM pada data uji dijelaskan oleh model; "
                "nilai R² dapat bernilai negatif jika prediksi lebih buruk daripada acuan rata-rata."
            )

            model = hasil["model"]
            st.markdown("#### Persamaan hasil model")
            st.latex(
                rf"\widehat{{IPM}} = {model.coef_[0]:.4f} \times RLS + ({model.intercept_:.4f})"
            )
            st.write(
                f"Koefisien RLS: **{format_angka(model.coef_[0], 4)}** · "
                f"Intersep: **{format_angka(model.intercept_, 4)}**"
            )

            data_uji = hasil["data_uji"].copy()
            data_uji = data_uji.loc[hasil["aktual"].index].copy()
            data_uji["IPM aktual"] = hasil["aktual"].to_numpy()
            data_uji["IPM prediksi"] = hasil["prediksi"]
            data_uji["Galat absolut"] = (
                data_uji["IPM aktual"] - data_uji["IPM prediksi"]
            ).abs()

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(hasil["aktual"], hasil["prediksi"], alpha=0.7)
            batas_min = min(float(np.min(hasil["aktual"])), float(np.min(hasil["prediksi"])))
            batas_max = max(float(np.max(hasil["aktual"])), float(np.max(hasil["prediksi"])))
            ax.plot([batas_min, batas_max], [batas_min, batas_max], linestyle="--")
            ax.set_xlabel("IPM aktual")
            ax.set_ylabel("IPM prediksi")
            ax.set_title(f"IPM Aktual vs Prediksi — {pilihan_regresi}")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("#### Rincian prediksi pada data uji")
            kolom_tampil = [
                c for c in ["Provinsi", "Kabupaten_Kota", "Nama_Wilayah", "Tahun", "RLS"]
                if c in data_uji.columns
            ] + ["IPM aktual", "IPM prediksi", "Galat absolut"]
            st.dataframe(
                data_uji[kolom_tampil].round(3),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Regresi belum dijalankan karena data valid belum mencukupi.")


# =========================================================
# 9. HALAMAN STATISTIK IPM
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

    df_tren = pd.DataFrame(tren).set_index("Tahun")

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
            "Tahun": TAHUN,
            "Rata-rata lama sekolah": [
                data_tampilan[f"RLS_{tahun}"].mean()
                for tahun in TAHUN
            ]
        }).set_index("Tahun")

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

    daftar_wilayah = sorted(
        data_tampilan["Nama_Wilayah"].dropna().unique().tolist()
    )

    if len(daftar_wilayah) < 2:
        st.info("Pilih cakupan yang memiliki setidaknya dua wilayah.")
    else:
        wilayah_a = st.selectbox(
            "Wilayah pertama",
            daftar_wilayah,
            index=0
        )
        pilihan_b = [w for w in daftar_wilayah if w != wilayah_a]
        wilayah_b = st.selectbox(
            "Wilayah kedua",
            pilihan_b,
            index=0
        )

        baris_a = data_tampilan[
            data_tampilan["Nama_Wilayah"] == wilayah_a
        ].iloc[0]

        baris_b = data_tampilan[
            data_tampilan["Nama_Wilayah"] == wilayah_b
        ].iloc[0]

        st.markdown("### Perbandingan pada tahun pilihan")

        perbandingan = pd.DataFrame({
            "Indikator": [
                "IPM laki-laki",
                "IPM perempuan",
                "Rata-rata lama sekolah"
            ],
            wilayah_a: [
                baris_a[kolom_l],
                baris_a[kolom_p],
                baris_a[kolom_rls]
            ],
            wilayah_b: [
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

        st.subheader("Perbandingan dari Tahun ke Tahun")

        tren_band = pd.DataFrame({
            "Tahun": TAHUN,
            f"{wilayah_a} - IPM Laki-laki": [
                baris_a[f"IPM_L_{tahun}"] for tahun in TAHUN
            ],
            f"{wilayah_b} - IPM Laki-laki": [
                baris_b[f"IPM_L_{tahun}"] for tahun in TAHUN
            ]
        }).set_index("Tahun")

        st.line_chart(tren_band)

        st.subheader("Posisi Wilayah dalam Data")

        for kolom, label in [
            (kolom_l, "IPM laki-laki"),
            (kolom_p, "IPM perempuan"),
            (kolom_rls, "rata-rata lama sekolah")
        ]:
            seluruh = data_tampilan[kolom].dropna().sort_values(
                ascending=False
            )

            if wilayah_a in data_tampilan["Nama_Wilayah"].values:
                nilai_a = baris_a[kolom]
                peringkat_a = (
                    int((seluruh > nilai_a).sum() + 1)
                    if pd.notna(nilai_a) else None
                )
            else:
                peringkat_a = None

            if wilayah_b in data_tampilan["Nama_Wilayah"].values:
                nilai_b = baris_b[kolom]
                peringkat_b = (
                    int((seluruh > nilai_b).sum() + 1)
                    if pd.notna(nilai_b) else None
                )
            else:
                peringkat_b = None

            if peringkat_a is not None and peringkat_b is not None:
                satuan = " tahun" if kolom == kolom_rls else " poin"
                kartu_temuan(
                    f"Posisi {label}",
                    f"{wilayah_a} berada pada peringkat {peringkat_a} "
                    f"dengan nilai {format_angka(baris_a[kolom])}{satuan}, "
                    f"sedangkan {wilayah_b} berada pada peringkat {peringkat_b} "
                    f"dengan nilai {format_angka(baris_b[kolom])}{satuan}."
                )


# =========================================================
# 11. POIN 5–7: PERSIAPAN DATA, KORELASI, DAN SUMBER DATA
# =========================================================

elif halaman == "Persiapan Data (5–7)":
    st.subheader("Poin 5 — Data Cleaning dan Normalisasi")
    st.write(
        "Bagian ini merangkum proses penanganan nilai yang tidak tersedia, "
        "duplikasi, dan normalisasi sebelum data digunakan dalam analisis."
    )

    a, b, c, d = st.columns(4)
    a.metric("Baris sebelum cleaning", f"{jumlah_baris_awal:,}")
    b.metric("Baris setelah cleaning", f"{jumlah_baris_bersih:,}")
    c.metric("Duplikasi identik dihapus", f"{jumlah_duplikat_awal:,}")
    d.metric("Nilai diperkirakan secara linier", f"{jumlah_diperkirakan:,}")

    st.markdown("#### Penanganan data tidak tersedia")
    st.markdown(
        "- **Interpolasi linier:** memperkirakan nilai kosong yang berada di antara dua tahun dengan data.\n"
        "- **Ekstrapolasi linier:** memperkirakan nilai kosong pada tahun sebelum atau sesudah rentang data yang tersedia.\n"
        "- Perkiraan dilakukan per wilayah dan per indikator. Jika suatu indikator memiliki kurang dari dua nilai tahun yang tersedia, nilai kosongnya dibiarkan kosong karena tren linier belum dapat dihitung secara memadai."
    )
    st.caption(
        f"Nilai kosong pada kolom numerik sebelum cleaning: {jumlah_kosong_awal:,}. "
        f"Nilai kosong yang masih tersisa setelah proses: {jumlah_kosong_setelah:,}. "
        "Nilai hasil interpolasi/ekstrapolasi merupakan estimasi, bukan angka asli dari sumber."
    )

    st.markdown("#### Normalisasi MinMaxScaler")
    st.write(
        "MinMaxScaler mengubah nilai fitur numerik ke rentang 0–1. "
        "Pada dashboard ini, hasil normalisasi digunakan untuk fitur K-Means, "
        "agar perbedaan rentang skala antarvariabel tidak mendominasi perhitungan jarak."
    )
    st.latex(r"X' = \frac{X-X_{min}}{X_{max}-X_{min}}")

    with st.expander("Lihat jumlah nilai kosong per variabel"):
        tabel_kosong = pd.DataFrame({
            "Variabel": KOLOM_NUMERIK,
            "Kosong setelah cleaning": [int(df[k].isna().sum()) for k in KOLOM_NUMERIK],
            "Data tersedia": [int(df[k].notna().sum()) for k in KOLOM_NUMERIK]
        })
        st.dataframe(tabel_kosong, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Poin 6 — Matriks Korelasi")
    tahun_korelasi = st.selectbox(
        "Pilih tahun untuk matriks korelasi", TAHUN,
        index=len(TAHUN) - 1, key="tahun_korelasi"
    )
    kolom_korelasi = [
        f"IPM_L_{tahun_korelasi}",
        f"IPM_P_{tahun_korelasi}",
        f"RLS_{tahun_korelasi}"
    ]
    nama_korelasi = {
        f"IPM_L_{tahun_korelasi}": "IPM Laki-laki",
        f"IPM_P_{tahun_korelasi}": "IPM Perempuan",
        f"RLS_{tahun_korelasi}": "RLS"
    }
    matriks = df[kolom_korelasi].rename(columns=nama_korelasi).corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    gambar = ax.imshow(matriks.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(matriks.columns)), matriks.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(matriks.index)), matriks.index)
    for i in range(len(matriks.index)):
        for j in range(len(matriks.columns)):
            nilai = matriks.iloc[i, j]
            ax.text(j, i, f"{nilai:.2f}", ha="center", va="center", color="black")
    ax.set_title(f"Matriks Korelasi Pearson Tahun {tahun_korelasi}")
    fig.colorbar(gambar, ax=ax, label="Koefisien korelasi")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Nilai mendekati +1 menunjukkan hubungan linear positif; mendekati -1 menunjukkan "
        "hubungan linear negatif; dan mendekati 0 menunjukkan hubungan linear yang lemah. "
        "Korelasi tidak membuktikan hubungan sebab-akibat."
    )
    st.dataframe(matriks.round(3), use_container_width=True)

    st.divider()
    st.subheader("Poin 7 — Dataset dan Sumber Data")
    st.write(
        "File Excel yang digunakan menggabungkan dua kelompok data/indikator, "
        "yaitu IPM berdasarkan jenis kelamin dan rata-rata lama sekolah (RLS), "
        "untuk wilayah dan tahun 2010–2013. Keduanya dianalisis bersama untuk "
        "mengkaji pola pembangunan manusia dan pendidikan."
    )
    st.markdown("**Dataset 1 — IPM berdasarkan jenis kelamin**")
    st.write("Variabel: IPM laki-laki dan IPM perempuan pada setiap wilayah, tahun 2010–2013.")
    st.markdown("Sumber BPS: [Indeks Pembangunan Manusia (IPM) menurut Jenis Kelamin](https://www.bps.go.id/id/statistics-table/2/NDYyIzI%3D/indeks-pembangunan-manusia--ipm--menurut-jenis-kelamin.html)")

    st.markdown("**Dataset 2 — Rata-rata Lama Sekolah (RLS)**")
    st.write("Variabel: rata-rata lama sekolah dalam satuan tahun pada setiap wilayah, tahun 2010–2013.")
    st.markdown("Sumber BPS: [Rata-rata Lama Sekolah Tahun](https://www.bps.go.id/id/statistics-table/2/NSMy/rata-rata-lama-sekolah-tahun.html)")
    st.warning(
        "Pastikan halaman atau publikasi BPS yang kamu cantumkan memang memuat data historis 2010–2013 "
        "yang digunakan dalam file Excel. Jika data berasal dari publikasi/unduhan lain, cantumkan sumber "
        "aslinya agar tahun dan cakupan data dapat diverifikasi."
    )


# =========================================================
# 12. HALAMAN TENTANG DATA
# =========================================================

elif halaman == "Tentang Data":
    st.subheader("Tentang Dataset")

    st.markdown("""
    Dashboard ini menggunakan data Indeks Pembangunan Manusia (IPM)
    berdasarkan jenis kelamin serta rata-rata lama sekolah (RLS).

    Data digunakan untuk menggambarkan distribusi indikator,
    membandingkan wilayah, mengamati perubahan tahunan, dan menelaah
    hubungan statistik antara pendidikan dan IPM.

    Nilai yang tidak tersedia diperkirakan menggunakan interpolasi atau ekstrapolasi linier
    apabila tersedia minimal dua titik data untuk menghitung tren. Nilai hasil perkiraan
    merupakan estimasi dan bukan nilai asli sumber.
    """)

    a, b, c = st.columns(3)
    a.metric("Jumlah baris", f"{len(df):,}")
    b.metric("Jumlah provinsi", f"{df['Provinsi'].nunique():,}")
    c.metric("Jumlah wilayah unik", f"{df['Nama_Wilayah'].nunique():,}")

    st.markdown("### Cakupan variabel")
    st.write(
        "- IPM laki-laki dan perempuan: 2010–2013\n"
        "- Rata-rata lama sekolah: 2010–2013\n"
        "- Identitas wilayah: provinsi dan kabupaten/kota"
    )

    st.markdown("### Kelengkapan data")

    tabel_kosong = pd.DataFrame({
        "Variabel": KOLOM_NUMERIK,
        "Jumlah data kosong": [
            int(df[k].isna().sum()) for k in KOLOM_NUMERIK
        ],
        "Jumlah data tersedia": [
            int(df[k].notna().sum()) for k in KOLOM_NUMERIK
        ]
    })

    st.dataframe(
        tabel_kosong,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Statistik deskriptif")
    st.dataframe(
        df[KOLOM_NUMERIK].describe().T,
        use_container_width=True
    )

    st.download_button(
        "Unduh ringkasan statistik (CSV)",
        data=df[KOLOM_NUMERIK].describe().T.to_csv().encode("utf-8-sig"),
        file_name="ringkasan_statistik.csv",
        mime="text/csv"
    )


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()
st.caption(
    "Dashboard Statistik Pembangunan Manusia dan Pendidikan · "
    "Sumber data bersumber Badan Pusat Statistik."
)