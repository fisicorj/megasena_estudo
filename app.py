import os
import json
import secrets
import hashlib
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# DEFAULT CONFIG
# =========================
DEFAULT_EXCEL_PATH = "Mega-Sena.xlsx"
DEFAULT_LOG_PATH = "megasena_geracoes.jsonl"

BOLAS_COLS = ["Bola1", "Bola2", "Bola3", "Bola4", "Bola5", "Bola6"]
DATA_COL = "Data do Sorteio"


# =========================
# UTIL
# =========================
def now_sp():
    return datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(timespec="seconds")

def bytes_hash_sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def file_hash_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save_log(record: dict, path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def fmt_nums(nums):
    return " ".join(f"{int(n):02d}" for n in nums)


# =========================
# LOAD
# =========================
def load_megasena_from_df(df: pd.DataFrame) -> pd.DataFrame:
    if DATA_COL in df.columns:
        df[DATA_COL] = pd.to_datetime(df[DATA_COL], dayfirst=True, errors="coerce")

    df = df.dropna(subset=BOLAS_COLS).copy()
    df[BOLAS_COLS] = df[BOLAS_COLS].astype(int)

    if DATA_COL in df.columns and df[DATA_COL].notna().any():
        df = df.sort_values(DATA_COL).reset_index(drop=True)
    return df

def load_megasena_from_upload(uploaded_file) -> tuple[pd.DataFrame, str, str]:
    data = uploaded_file.getvalue()
    sha = bytes_hash_sha256(data)
    bio = BytesIO(data)
    df = pd.read_excel(bio, engine="openpyxl")
    return load_megasena_from_df(df), sha, uploaded_file.name

def load_megasena_from_path(path: str) -> tuple[pd.DataFrame, str, str]:
    sha = file_hash_sha256(path)
    df = pd.read_excel(path, engine="openpyxl")
    return load_megasena_from_df(df), sha, path


# =========================
# TOPS (pos + geral)
# =========================
def top_by_position_with_pct(df: pd.DataFrame, topk: int = 10):
    out = {}
    total = len(df)
    for col in BOLAS_COLS:
        counts = df[col].value_counts().sort_values(ascending=False)
        items = []
        for n, c in zip(counts.index[:topk], counts.values[:topk]):
            pct = (c / total) * 100.0
            items.append({"n": int(n), "count": int(c), "pct": float(pct)})
        out[col] = items
    return out

def top_overall(df: pd.DataFrame, topk: int = 15):
    all_nums = df[BOLAS_COLS].to_numpy().reshape(-1)
    counts = Counter(all_nums)
    total_bolas = len(all_nums)
    items = []
    for n, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:topk]:
        pct = (c / total_bolas) * 100.0
        items.append({"n": int(n), "count": int(c), "pct": float(pct)})
    return items


# =========================
# PESOS (freq + recência)
# =========================
def build_weights(df: pd.DataFrame, w_freq: float, w_recency: float, recency_lambda: float) -> np.ndarray:
    all_nums = df[BOLAS_COLS].to_numpy().reshape(-1)

    freq = Counter(all_nums)
    freq_arr = np.zeros(61, dtype=float)
    for n in range(1, 61):
        freq_arr[n] = freq.get(n, 0)
    freq_arr[1:] += 1.0
    freq_arr[1:] /= freq_arr[1:].sum()

    rec_arr = np.zeros(61, dtype=float)
    if (DATA_COL in df.columns) and df[DATA_COL].notna().any():
        last_date = df[DATA_COL].max()
        ages = (last_date - df[DATA_COL]).dt.days.fillna(0).to_numpy()
    else:
        ages = (len(df) - 1 - np.arange(len(df))).astype(float)

    draw_w = np.exp(-recency_lambda * ages)
    bolas = df[BOLAS_COLS].to_numpy()
    for i in range(len(df)):
        for n in bolas[i]:
            rec_arr[n] += draw_w[i]

    rec_arr[1:] += 1e-9
    rec_arr[1:] /= rec_arr[1:].sum()

    p = w_freq * freq_arr + w_recency * rec_arr
    p[0] = 0.0
    p[1:] /= p[1:].sum()
    return p


# =========================
# HEURÍSTICAS
# =========================
def count_consecutivos(nums_sorted):
    c = 0
    best = 0
    for i in range(1, len(nums_sorted)):
        if nums_sorted[i] == nums_sorted[i - 1] + 1:
            c += 1
            best = max(best, c)
        else:
            c = 0
    return best

def dezenas_group(n):
    return (n - 1) // 10

def is_good_game(nums, min_impares, max_impares, max_consecutivos, max_por_dezena):
    nums = sorted(nums)
    odd = sum(n % 2 == 1 for n in nums)
    if not (min_impares <= odd <= max_impares):
        return False
    if count_consecutivos(nums) > max_consecutivos:
        return False
    groups = Counter(dezenas_group(n) for n in nums)
    if any(v > max_por_dezena for v in groups.values()):
        return False
    return True


# =========================
# POOL + SCORE
# =========================
def game_score(nums, p):
    eps = 1e-12
    return float(np.sum(np.log(np.clip(p[list(nums)], eps, 1.0))))

def generate_pool_with_progress(
    p,
    pool_size,
    seed_run,
    heur_min_impares,
    heur_max_impares,
    heur_max_consecutivos,
    heur_max_por_dezena,
    progress_cb=None,
):
    rng = np.random.default_rng(seed_run)
    seen = set()
    pool = []

    hard_limit = pool_size * 40
    tries = 0

    # atualiza progress por tentativas (porque pool é filtrado por heurísticas)
    step = max(1, hard_limit // 200)

    while len(pool) < pool_size and tries < hard_limit:
        tries += 1

        nums = tuple(sorted(map(int, rng.choice(np.arange(1, 61), size=6, replace=False, p=p[1:]))))
        if nums in seen:
            if progress_cb and tries % step == 0:
                progress_cb(tries, hard_limit, len(pool))
            continue
        seen.add(nums)

        if not is_good_game(
            nums,
            min_impares=heur_min_impares,
            max_impares=heur_max_impares,
            max_consecutivos=heur_max_consecutivos,
            max_por_dezena=heur_max_por_dezena,
        ):
            if progress_cb and tries % step == 0:
                progress_cb(tries, hard_limit, len(pool))
            continue

        pool.append(nums)

        if progress_cb and tries % step == 0:
            progress_cb(tries, hard_limit, len(pool))

    if progress_cb:
        progress_cb(hard_limit, hard_limit, len(pool))

    return pool, tries


# =========================
# 20 COM COBERTURA
# =========================
def select_coverage_games(pool, p, n_games: int, beta: float, scan_top: int = 50_000):
    scored = [(game_score(nums, p), nums) for nums in pool]
    scored.sort(reverse=True, key=lambda x: x[0])

    used = Counter()
    chosen = []
    chosen_set = set()

    scan_top = min(scan_top, len(scored))

    for _ in range(n_games):
        best_val = -1e18
        best_nums = None

        for base_s, nums in scored[:scan_top]:
            if nums in chosen_set:
                continue
            rep_penalty = sum(used[n] for n in nums)
            val = base_s - beta * rep_penalty
            if val > best_val:
                best_val = val
                best_nums = nums

        if best_nums is None:
            break

        chosen.append(list(best_nums))
        chosen_set.add(best_nums)
        for n in best_nums:
            used[n] += 1

    return chosen, used, scored, chosen_set


# =========================
# + TOP SEM ESTRAGAR
# =========================
def select_top_without_ruining(
    scored,
    chosen_set,
    used_counts,
    k: int,
    max_count_per_number_total: int,
    max_overlap_with_coverage: int,
    dont_increase_peak: bool,
):
    top_games = []
    used_total = used_counts.copy()
    current_peak = max(used_total.values()) if used_total else 0

    for base_s, nums in scored:
        if len(top_games) >= k:
            break
        if nums in chosen_set:
            continue

        nums_list = list(nums)
        overlap = sum(1 for n in nums_list if used_total[n] > 0)
        if overlap > max_overlap_with_coverage:
            continue

        new_peak = current_peak
        ok = True
        for n in nums_list:
            new_count = used_total[n] + 1
            if new_count > max_count_per_number_total:
                ok = False
                break
            new_peak = max(new_peak, new_count)

        if not ok:
            continue
        if dont_increase_peak and new_peak > current_peak:
            continue

        top_games.append({"nums": nums_list, "score": float(base_s), "overlap_with_20": int(overlap)})
        chosen_set.add(nums)
        for n in nums_list:
            used_total[n] += 1
        current_peak = max(current_peak, new_peak)

    return top_games, used_total


# =========================
# METRICS
# =========================
def coverage_metrics(used_counts: Counter):
    if not used_counts:
        return {"peak": 0, "distinct": 0, "most_common": []}
    mc = used_counts.most_common(10)
    return {
        "peak": mc[0][1],
        "distinct": len(used_counts),
        "most_common": [{"n": int(n), "count": int(c)} for n, c in mc],
    }


# =========================
# UI helpers (visual)
# =========================
def inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { padding: 10px 14px; border-radius: 12px; }
        .card {
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.02);
            border-radius: 16px;
            padding: 16px;
        }
        .muted { opacity: 0.75; }
        code { border-radius: 8px; padding: 2px 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def style_top_table(df):
    # df com colunas: n, count, pct (pct já pode estar em float)
    styler = df.style.format(
        {
            "n": lambda x: f"{int(x):02d}",
            "count": "{:.0f}",
            "pct": "{:.2f}%",
        }
    )
    styler = styler.background_gradient(subset=["count"], cmap="Blues")
    styler = styler.background_gradient(subset=["pct"], cmap="Greens")
    return styler


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Mega-Sena • Cobertura + Score", layout="wide")
inject_css()

st.markdown("## 🎲 Mega-Sena — Gerador de jogos (cobertura + top score)")
st.markdown('<div class="muted">Carregue seu Excel, ajuste parâmetros e gere jogos com seed registrada (reprodutível).</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📥 Entrada")
    up = st.file_uploader("Upload do Excel (.xlsx)", type=["xlsx"])

    st.divider()
    st.header("⚙️ Execução")
    run_btn = st.button("🚀 Gerar jogos", type="primary", use_container_width=True)

    seed_mode = st.radio("Seed", ["Aleatória", "Fixar"], horizontal=True)
    seed_fixed = None
    if seed_mode == "Fixar":
        seed_fixed = st.number_input("Seed fixa (uint64)", min_value=0, max_value=2**64 - 1, value=123, step=1)

    st.divider()
    st.header("🧠 Parâmetros (principal)")
    pool_size = st.number_input("POOL_SIZE", min_value=10_000, max_value=2_000_000, value=200_000, step=10_000)
    scan_top = st.number_input("SCAN_TOP (cobertura)", min_value=5_000, max_value=300_000, value=50_000, step=5_000)
    n_cobertura = st.number_input("N_COBERTURA (jogos)", min_value=5, max_value=50, value=20, step=1)
    n_top = st.number_input("N_TOP (top jogos)", min_value=0, max_value=10, value=2, step=1)

    st.divider()
    st.header("📌 Pesos (freq + recência)")
    w_freq = st.slider("W_FREQ", 0.0, 1.0, 0.55, 0.01)
    w_recency = st.slider("W_RECENCY", 0.0, 1.0, 0.45, 0.01)
    recency_lambda = st.number_input("RECENCY_LAMBDA", min_value=0.0, max_value=0.05, value=0.004, step=0.001, format="%.3f")

    st.divider()
    st.header("🧩 Heurísticas")
    max_consecutivos = st.number_input("MAX_CONSECUTIVOS", 0, 5, 2, 1)
    min_impares = st.number_input("MIN_IMPARES", 0, 6, 2, 1)
    max_impares = st.number_input("MAX_IMPARES", 0, 6, 4, 1)
    max_por_dezena = st.number_input("MAX_POR_DEZENA", 1, 6, 3, 1)

    st.divider()
    st.header("🛡️ Cobertura / TOP rules")
    cover_beta = st.slider("COVER_BETA", 0.0, 2.0, 0.45, 0.01)
    max_count_total = st.number_input("MAX_COUNT_PER_NUMBER_TOTAL", 1, 10, 3, 1)
    max_overlap = st.number_input("MAX_OVERLAP_WITH_COVERAGE", 0, 6, 4, 1)
    dont_increase_peak = st.toggle("DONT_INCREASE_PEAK", value=True)

    st.divider()
    st.header("🧑‍🏫 Modo didático")
    modo_didatico = st.toggle("Mostrar avisos e explicações", value=False)

    st.divider()
    st.header("🧾 Log")
    enable_log = st.toggle("Salvar log JSONL", value=False)
    log_path = st.text_input("LOG_PATH", value=DEFAULT_LOG_PATH, disabled=not enable_log)

# Load dataset
excel_sha = None
excel_name = None

try:
    if up is not None:
        df, excel_sha, excel_name = load_megasena_from_upload(up)
    else:
        if os.path.exists(DEFAULT_EXCEL_PATH):
            df, excel_sha, excel_name = load_megasena_from_path(DEFAULT_EXCEL_PATH)
        else:
            df = None
except Exception as e:
    st.error(f"Erro ao carregar o Excel: {e}")
    st.stop()

if df is None:
    st.info("Envie um arquivo Excel (.xlsx) ou coloque `Mega-Sena.xlsx` na mesma pasta do app.")
    st.stop()

# Didactic box
if modo_didatico:
    st.info(
        "📌 **Observação didática**: frequência histórica e recência **não são previsão**. "
        "O algoritmo é um estudo estatístico/heurístico para **gerar combinações** com critérios de cobertura, "
        "mas sorteio é aleatório."
    )

# Tabs
tab_stats, tab_run, tab_export, tab_debug = st.tabs(["📊 Estatísticas", "✅ Jogos", "⬇️ Exportar", "🧪 Debug"])

# Prévia
with tab_debug:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(f"Arquivo: **{excel_name}**")
    st.write(f"SHA256: `{excel_sha}`")
    st.write(f"Linhas válidas: **{len(df)}**")
    with st.expander("Prévia dos dados (15 primeiras linhas)", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Estatísticas (sempre)
topk_pos = 10
topk_geral = 15
top_pos = top_by_position_with_pct(df, topk=topk_pos)
top_all = top_overall(df, topk=topk_geral)

with tab_stats:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top por posição (contagem e %)")
        for col in BOLAS_COLS:
            tdf = pd.DataFrame(top_pos[col])
            st.markdown(f"**{col}**")
            st.dataframe(style_top_table(tdf), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top geral (contagem e %)")
        tdf = pd.DataFrame(top_all)
        st.dataframe(style_top_table(tdf), use_container_width=True, hide_index=True)

        # gráfico simples sem dependências extras
        chart_df = tdf.copy()
        chart_df["n"] = chart_df["n"].map(lambda x: f"{int(x):02d}")
        chart_df = chart_df.set_index("n")[["count"]]
        st.caption("Gráfico (Top 15 por frequência):")
        st.bar_chart(chart_df)

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# EXECUÇÃO: só roda ao clicar no botão
# =========================
def normalize_weights(wf, wr):
    if abs((wf + wr) - 1.0) > 1e-9:
        s = wf + wr
        if s <= 0:
            return 0.5, 0.5
        wf = wf / s
        wr = 1.0 - wf
    return wf, wr

def make_key(seed_run, excel_sha, params: dict):
    # key estável p/ session cache
    base = json.dumps({"seed": seed_run, "sha": excel_sha, "params": params}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(base).hexdigest()

def log_allowed_on_public() -> bool:
    # heurística: se estiver no streamlit cloud, desativa por padrão
    # (você pode adaptar isso ao seu ambiente)
    if os.getenv("STREAMLIT_CLOUD") or os.getenv("STREAMLIT_RUNTIME"):
        return False
    return True

with tab_run:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Geração de jogos")
    st.caption("A geração (pool + seleção) só acontece quando você clica em **Gerar jogos** na sidebar.")

    if run_btn:
        # seed
        seed_run = int(seed_fixed) if seed_mode == "Fixar" else secrets.randbits(64)

        # parâmetros normalizados
        w_freq_n, w_recency_n = normalize_weights(w_freq, w_recency)

        params = {
            "pool_size": int(pool_size),
            "scan_top": int(scan_top),
            "n_cobertura": int(n_cobertura),
            "n_top": int(n_top),
            "recency_lambda": float(recency_lambda),
            "w_freq": float(w_freq_n),
            "w_recency": float(w_recency_n),
            "cover_beta": float(cover_beta),
            "heuristics": {
                "max_consecutivos": int(max_consecutivos),
                "min_impares": int(min_impares),
                "max_impares": int(max_impares),
                "max_por_dezena": int(max_por_dezena),
            },
            "top_rules": {
                "max_count_total": int(max_count_total),
                "max_overlap": int(max_overlap),
                "dont_increase_peak": bool(dont_increase_peak),
            },
        }

        run_key = make_key(seed_run, excel_sha, params)
        st.session_state["last_run_key"] = run_key

        # Se já existe em cache (session_state), reutiliza
        if "runs" not in st.session_state:
            st.session_state["runs"] = {}

        if run_key in st.session_state["runs"]:
            st.success(f"✅ Resultado reutilizado do cache (sessão). seed={seed_run}")
            result = st.session_state["runs"][run_key]
        else:
            st.success(f"Execução iniciada: {now_sp()} | seed={seed_run}")

            # pesos
            with st.spinner("Calculando pesos (freq + recência)..."):
                p = build_weights(df, w_freq=w_freq_n, w_recency=w_recency_n, recency_lambda=float(recency_lambda))

            # pool com progresso real
            prog = st.progress(0.0)
            status = st.empty()

            def progress_cb(tries, hard_limit, pool_len):
                frac = min(tries / max(1, hard_limit), 1.0)
                prog.progress(frac)
                status.write(f"Tentativas: **{tries:,}** / {hard_limit:,} • Pool válido: **{pool_len:,}**")

            with st.spinner(f"Gerando pool (até {int(pool_size):,}) com heurísticas..."):
                pool, tries = generate_pool_with_progress(
                    p=p,
                    pool_size=int(pool_size),
                    seed_run=seed_run,
                    heur_min_impares=int(min_impares),
                    heur_max_impares=int(max_impares),
                    heur_max_consecutivos=int(max_consecutivos),
                    heur_max_por_dezena=int(max_por_dezena),
                    progress_cb=progress_cb,
                )

            prog.progress(1.0)
            status.write(f"Pool final: **{len(pool):,}** (tentativas: {tries:,})")

            # seleção
            with st.spinner("Selecionando jogos com cobertura + TOPs..."):
                jogos_20, used_20, scored, chosen_set = select_coverage_games(
                    pool, p, n_games=int(n_cobertura), beta=float(cover_beta), scan_top=int(scan_top)
                )
                top_games, used_total = select_top_without_ruining(
                    scored=scored,
                    chosen_set=chosen_set,
                    used_counts=used_20,
                    k=int(n_top),
                    max_count_per_number_total=int(max_count_total),
                    max_overlap_with_coverage=int(max_overlap),
                    dont_increase_peak=bool(dont_increase_peak),
                )

            m20 = coverage_metrics(used_20)
            mt = coverage_metrics(used_total)

            result = {
                "timestamp_sp": now_sp(),
                "seed": seed_run,
                "excel_name": excel_name,
                "excel_sha256": excel_sha,
                "rows": int(len(df)),
                "params": params,
                "top_position": top_pos,
                "top_overall": top_all,
                "pool_len": int(len(pool)),
                "pool_tries": int(tries),
                "games_coverage": jogos_20,
                "games_top": top_games,
                "coverage_metrics_20": m20,
                "coverage_metrics_total": mt,
            }

            # guarda cache em sessão
            st.session_state["runs"][run_key] = result

            # log (controlado)
            if enable_log:
                if log_allowed_on_public():
                    try:
                        save_log(result, log_path)
                        st.info(f"Log salvo em: `{log_path}`")
                    except Exception as e:
                        st.warning(f"Não consegui salvar log em `{log_path}`: {e}")
                else:
                    st.warning("Log desativado automaticamente em ambiente público/cloud (segurança).")

        # Exibir resultado
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Seed", str(result["seed"]))
        c2.metric("Pool válido", f"{result['pool_len']:,}")
        c3.metric("Distintos (20)", result["coverage_metrics_20"]["distinct"])
        c4.metric("Peak (20)", result["coverage_metrics_20"]["peak"])

        st.divider()

        # 20 jogos
        st.markdown("### ✅ Jogos com cobertura")
        jogos_20 = result["games_coverage"]
        jogos_df = pd.DataFrame(jogos_20, columns=[f"N{i}" for i in range(1, 7)])
        st.dataframe(jogos_df, use_container_width=True, hide_index=True)

        # Top
        st.markdown("### 🏆 TOP (sem estragar a cobertura)")
        top_games = result["games_top"]
        if len(top_games) == 0 and result["params"]["n_top"] > 0:
            st.warning("Não encontrei TOPs que respeitem as restrições. Afrouxe os limites.")
        else:
            top_df = pd.DataFrame(
                [
                    {
                        "Jogo": fmt_nums(t["nums"]),
                        "Score": round(float(t["score"]), 4),
                        "Overlap com 20": f"{int(t['overlap_with_20'])}/6",
                    }
                    for t in top_games
                ]
            )
            st.dataframe(top_df, use_container_width=True, hide_index=True)

        st.divider()

        # Métricas
        st.markdown("### 📊 Métricas")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**20 jogos**")
            st.json(result["coverage_metrics_20"])
        with mc2:
            st.markdown("**Total (20 + TOP)**")
            st.json(result["coverage_metrics_total"])

    else:
        st.info("Clique em **Gerar jogos** na sidebar para rodar. Os resultados ficam salvos no cache da sessão.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# EXPORT
# =========================
with tab_export:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⬇️ Exportar resultados")

    last_key = st.session_state.get("last_run_key")
    runs = st.session_state.get("runs", {})
    if not last_key or last_key not in runs:
        st.info("Nenhum resultado para exportar ainda. Rode uma geração na aba **Jogos**.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    result = runs[last_key]

    jogos_df = pd.DataFrame(result["games_coverage"], columns=[f"N{i}" for i in range(1, 7)])

    top_games = result["games_top"]
    top_df = pd.DataFrame(
        [
            {
                "Jogo": fmt_nums(t["nums"]),
                "Score": round(float(t["score"]), 4),
                "Overlap com 20": f"{int(t['overlap_with_20'])}/6",
            }
            for t in top_games
        ]
    ) if len(top_games) else pd.DataFrame(columns=["Jogo", "Score", "Overlap com 20"])

    # JSON
    st.download_button(
        "Baixar JSON do resultado",
        data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"megasena_result_{result['seed']}.json",
        mime="application/json",
        use_container_width=True,
    )

    # CSV (20 jogos)
    st.download_button(
        "Baixar CSV (20 jogos)",
        data=jogos_df.to_csv(index=False).encode("utf-8"),
        file_name=f"megasena_20_{result['seed']}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Excel com abas
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        # abas de jogos
        jogos_df.to_excel(writer, sheet_name="20_Jogos", index=False)
        top_df.to_excel(writer, sheet_name="Top", index=False)

        # estatísticas
        pd.DataFrame(result["top_overall"]).to_excel(writer, sheet_name="Top_Geral", index=False)

        # Top por posição (uma aba por posição fica chato; então concat em uma só)
        rows = []
        for col in BOLAS_COLS:
            for item in result["top_position"][col]:
                rows.append({"posicao": col, **item})
        pd.DataFrame(rows).to_excel(writer, sheet_name="Top_Posicao", index=False)

        # métricas
        pd.DataFrame([result["coverage_metrics_20"]]).to_excel(writer, sheet_name="Metricas_20", index=False)
        pd.DataFrame([result["coverage_metrics_total"]]).to_excel(writer, sheet_name="Metricas_Total", index=False)

        # params
        pd.json_normalize(result["params"]).to_excel(writer, sheet_name="Params", index=False)

    st.download_button(
        "Baixar Excel completo (abas)",
        data=out.getvalue(),
        file_name=f"megasena_{result['seed']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.caption("Dica: o Excel exportado já vem com abas de jogos, TOP, estatísticas e parâmetros.")
    st.markdown("</div>", unsafe_allow_html=True)
