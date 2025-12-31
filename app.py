import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import secrets
import hashlib
import os

# =========================
# DEFAULT CONFIG (pode ajustar na sidebar)
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
    # tenta criar diretório se existir
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

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
    df = pd.read_excel(uploaded_file)
    return load_megasena_from_df(df), sha, uploaded_file.name

def load_megasena_from_path(path: str) -> tuple[pd.DataFrame, str, str]:
    sha = file_hash_sha256(path)
    df = pd.read_excel(path)
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
def build_weights(df: pd.DataFrame,
                  w_freq: float,
                  w_recency: float,
                  recency_lambda: float) -> np.ndarray:
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
        if nums_sorted[i] == nums_sorted[i-1] + 1:
            c += 1
            best = max(best, c)
        else:
            c = 0
    return best

def dezenas_group(n):
    return (n - 1) // 10

def is_good_game(nums,
                 min_impares: int,
                 max_impares: int,
                 max_consecutivos: int,
                 max_por_dezena: int):
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

def generate_pool(p,
                  pool_size: int,
                  seed_run: int,
                  heur_min_impares: int,
                  heur_max_impares: int,
                  heur_max_consecutivos: int,
                  heur_max_por_dezena: int):
    rng = np.random.default_rng(seed_run)
    seen = set()
    pool = []
    # evita loop infinito em configs muito restritivas
    hard_limit = pool_size * 40
    tries = 0

    while len(pool) < pool_size and tries < hard_limit:
        tries += 1
        nums = tuple(sorted(map(int, rng.choice(np.arange(1, 61), size=6, replace=False, p=p[1:]))))
        if nums in seen:
            continue
        seen.add(nums)
        if not is_good_game(nums,
                            min_impares=heur_min_impares,
                            max_impares=heur_max_impares,
                            max_consecutivos=heur_max_consecutivos,
                            max_por_dezena=heur_max_por_dezena):
            continue
        pool.append(nums)

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

    for _ in range(n_games):
        best_val = -1e18
        best_nums = None

        for base_s, nums in scored[:min(scan_top, len(scored))]:
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
# +2 TOP SEM ESTRAGAR
# =========================
def select_top_without_ruining(scored, chosen_set, used_counts,
                               k: int,
                               max_count_per_number_total: int,
                               max_overlap_with_coverage: int,
                               dont_increase_peak: bool):
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
        "most_common": [{"n": int(n), "count": int(c)} for n, c in mc]
    }

# =========================
# CACHE (pool e seleção são pesados)
# =========================
@st.cache_data(show_spinner=False)
def cached_pool(p_tuple,
                pool_size,
                seed_run,
                heur_min_impares,
                heur_max_impares,
                heur_max_consecutivos,
                heur_max_por_dezena):
    p = np.array(p_tuple, dtype=float)
    pool, tries = generate_pool(
        p=p,
        pool_size=pool_size,
        seed_run=seed_run,
        heur_min_impares=heur_min_impares,
        heur_max_impares=heur_max_impares,
        heur_max_consecutivos=heur_max_consecutivos,
        heur_max_por_dezena=heur_max_por_dezena
    )
    return pool, tries

@st.cache_data(show_spinner=False)
def cached_selection(pool,
                     p_tuple,
                     n_cobertura,
                     cover_beta,
                     scan_top,
                     n_top,
                     max_count_total,
                     max_overlap,
                     dont_increase_peak):
    p = np.array(p_tuple, dtype=float)
    jogos_20, used_20, scored, chosen_set = select_coverage_games(
        pool, p, n_games=n_cobertura, beta=cover_beta, scan_top=scan_top
    )
    top2, used_total = select_top_without_ruining(
        scored=scored,
        chosen_set=chosen_set,
        used_counts=used_20,
        k=n_top,
        max_count_per_number_total=max_count_total,
        max_overlap_with_coverage=max_overlap,
        dont_increase_peak=dont_increase_peak
    )
    return jogos_20, used_20, scored, top2, used_total

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Mega-Sena - Gerador com Cobertura", layout="wide")
st.title("🎲 Mega-Sena — Gerador de jogos (cobertura + top score)")
st.caption("Carregue seu Excel, ajuste parâmetros e gere jogos com seed registrada (reprodutível).")

with st.sidebar:
    st.header("Entrada")
    up = st.file_uploader("Upload do Excel (.xlsx)", type=["xlsx"])

    st.divider()
    st.header("Parâmetros principais")
    pool_size = st.number_input("POOL_SIZE", min_value=10_000, max_value=2_000_000, value=200_000, step=10_000)
    scan_top = st.number_input("SCAN_TOP (cobertura)", min_value=5_000, max_value=300_000, value=50_000, step=5_000)

    n_cobertura = st.number_input("N_COBERTURA (jogos)", min_value=5, max_value=50, value=20, step=1)
    n_top = st.number_input("N_TOP (top jogos)", min_value=0, max_value=10, value=2, step=1)

    st.divider()
    st.header("Pesos histórico")
    w_freq = st.slider("W_FREQ", 0.0, 1.0, 0.55, 0.01)
    w_recency = st.slider("W_RECENCY", 0.0, 1.0, 0.45, 0.01)
    recency_lambda = st.number_input("RECENCY_LAMBDA", min_value=0.0, max_value=0.05, value=0.004, step=0.001, format="%.3f")

    st.divider()
    st.header("Heurísticas")
    max_consecutivos = st.number_input("MAX_CONSECUTIVOS", 0, 5, 2, 1)
    min_impares = st.number_input("MIN_IMPARES", 0, 6, 2, 1)
    max_impares = st.number_input("MAX_IMPARES", 0, 6, 4, 1)
    max_por_dezena = st.number_input("MAX_POR_DEZENA", 1, 6, 3, 1)

    st.divider()
    st.header("Cobertura / TOP rules")
    cover_beta = st.slider("COVER_BETA", 0.0, 2.0, 0.45, 0.01)
    max_count_total = st.number_input("MAX_COUNT_PER_NUMBER_TOTAL", 1, 10, 3, 1)
    max_overlap = st.number_input("MAX_OVERLAP_WITH_COVERAGE", 0, 6, 4, 1)
    dont_increase_peak = st.toggle("DONT_INCREASE_PEAK", value=True)

    st.divider()
    st.header("Seed & Log")
    seed_mode = st.radio("Seed", ["Aleatória", "Fixar"], horizontal=True)
    seed_fixed = st.number_input("Seed fixa (uint64)", min_value=0, max_value=2**64 - 1, value=123, step=1) if seed_mode == "Fixar" else None

    enable_log = st.toggle("Salvar log JSONL", value=False)
    log_path = st.text_input("LOG_PATH", value=DEFAULT_LOG_PATH, disabled=not enable_log)

    run_btn = st.button("🚀 Gerar jogos", type="primary")

# =========================
# LOAD DATA
# =========================
excel_sha = None
excel_name = None

if up is not None:
    df, excel_sha, excel_name = load_megasena_from_upload(up)
else:
    if os.path.exists(DEFAULT_EXCEL_PATH):
        df, excel_sha, excel_name = load_megasena_from_path(DEFAULT_EXCEL_PATH)
    else:
        df = None

if df is None:
    st.info("Envie um arquivo Excel (.xlsx) ou coloque `Mega-Sena.xlsx` na mesma pasta do app.")
    st.stop()

# =========================
# PREVIEW
# =========================
with st.expander("📄 Prévia dos dados (primeiras linhas)", expanded=False):
    st.write(f"Arquivo: **{excel_name}**  | SHA256: `{excel_sha}`  | Linhas: **{len(df)}**")
    st.dataframe(df.head(15), use_container_width=True)

# =========================
# STATS (rápido)
# =========================
topk_pos = 10
topk_geral = 15
top_pos = top_by_position_with_pct(df, topk=topk_pos)
top_all = top_overall(df, topk=topk_geral)

colA, colB = st.columns(2, gap="large")

with colA:
    st.subheader("📌 Top por posição")
    for col in BOLAS_COLS:
        tdf = pd.DataFrame(top_pos[col])
        tdf["n"] = tdf["n"].map(lambda x: f"{x:02d}")
        tdf["pct"] = tdf["pct"].map(lambda x: f"{x:.2f}%")
        st.markdown(f"**{col}**")
        st.dataframe(tdf, use_container_width=True, hide_index=True)

with colB:
    st.subheader("📌 Top geral")
    tdf = pd.DataFrame(top_all)
    tdf["n"] = tdf["n"].map(lambda x: f"{x:02d}")
    tdf["pct"] = tdf["pct"].map(lambda x: f"{x:.2f}%")
    st.dataframe(tdf, use_container_width=True, hide_index=True)

st.divider()

# =========================
# RUN GENERATION
# =========================
if run_btn:
    # validações simples
    if abs((w_freq + w_recency) - 1.0) > 1e-6:
        st.warning("W_FREQ + W_RECENCY deveria somar 1. Ajustei automaticamente.")
        s = w_freq + w_recency
        w_freq = w_freq / s if s > 0 else 0.5
        w_recency = 1.0 - w_freq

    seed_run = int(seed_fixed) if seed_mode == "Fixar" else secrets.randbits(64)

    st.success(f"Execução: {now_sp()} | seed={seed_run}")

    with st.spinner("Calculando pesos (freq + recência)..."):
        p = build_weights(df, w_freq=w_freq, w_recency=w_recency, recency_lambda=recency_lambda)
        p_tuple = tuple(float(x) for x in p)

    with st.spinner(f"Gerando pool (até {pool_size:,}) com heurísticas..."):
        pool, tries = cached_pool(
            p_tuple=p_tuple,
            pool_size=int(pool_size),
            seed_run=seed_run,
            heur_min_impares=int(min_impares),
            heur_max_impares=int(max_impares),
            heur_max_consecutivos=int(max_consecutivos),
            heur_max_por_dezena=int(max_por_dezena),
        )

    st.write(f"Pool gerado: **{len(pool):,}** jogos válidos (tentativas: {tries:,}).")

    with st.spinner("Selecionando 20 jogos com cobertura + TOPs..."):
        jogos_20, used_20, scored, top2, used_total = cached_selection(
            pool=pool,
            p_tuple=p_tuple,
            n_cobertura=int(n_cobertura),
            cover_beta=float(cover_beta),
            scan_top=int(scan_top),
            n_top=int(n_top),
            max_count_total=int(max_count_total),
            max_overlap=int(max_overlap),
            dont_increase_peak=bool(dont_increase_peak),
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Jogos de cobertura", len(jogos_20))
    with c2:
        st.metric("Números distintos (cobertura)", coverage_metrics(used_20)["distinct"])
    with c3:
        st.metric("Peak (cobertura)", coverage_metrics(used_20)["peak"])

    st.subheader(f"✅ {len(jogos_20)} jogos com COBERTURA")
    jogos_df = pd.DataFrame(jogos_20, columns=[f"N{i}" for i in range(1, 7)])
    st.dataframe(jogos_df, use_container_width=True, hide_index=True)

    st.subheader(f"🏆 {len(top2)} jogos TOP score (sem estragar a cobertura)")
    if len(top2) == 0 and n_top > 0:
        st.warning("Não encontrei TOPs que respeitem as restrições. Afrouxe os limites.")
    else:
        top_df = pd.DataFrame([{
            "Jogo": t["nums"],
            "Score": round(t["score"], 4),
            "Overlap com 20": f"{t['overlap_with_20']}/6"
        } for t in top2])
        st.dataframe(top_df, use_container_width=True, hide_index=True)

    st.subheader("📊 Métricas de cobertura")
    m20 = coverage_metrics(used_20)
    mt = coverage_metrics(used_total)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**20 jogos**")
        st.json(m20)
    with mc2:
        st.markdown("**Total (20 + TOP)**")
        st.json(mt)

    # =========================
    # EXPORTS
    # =========================
    st.subheader("⬇️ Exportar resultados")
    out_payload = {
        "timestamp_sp": now_sp(),
        "seed": seed_run,
        "excel_name": excel_name,
        "excel_sha256": excel_sha,
        "rows": int(len(df)),
        "params": {
            "pool_size": int(pool_size),
            "scan_top": int(scan_top),
            "recency_lambda": float(recency_lambda),
            "w_freq": float(w_freq),
            "w_recency": float(w_recency),
            "coverage_beta": float(cover_beta),
            "heuristics": {
                "max_consecutivos": int(max_consecutivos),
                "min_impares": int(min_impares),
                "max_impares": int(max_impares),
                "max_por_dezena": int(max_por_dezena),
            },
            "top_rules": {
                "max_count_per_number_total": int(max_count_total),
                "max_overlap_with_coverage": int(max_overlap),
                "dont_increase_peak": bool(dont_increase_peak),
            }
        },
        "top_position": top_pos,
        "top_overall": top_all,
        "games_coverage": jogos_20,
        "games_top": top2,
        "coverage_metrics_20": m20,
        "coverage_metrics_total": mt,
    }

    st.download_button(
        "Baixar JSON do resultado",
        data=json.dumps(out_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"megasena_result_{seed_run}.json",
        mime="application/json",
    )

    st.download_button(
        "Baixar CSV (20 jogos)",
        data=jogos_df.to_csv(index=False).encode("utf-8"),
        file_name=f"megasena_20_{seed_run}.csv",
        mime="text/csv",
    )

    # =========================
    # LOG
    # =========================
    if enable_log:
        try:
            save_log(out_payload, log_path)
            st.info(f"Log salvo em: `{log_path}`")
        except Exception as e:
            st.error(f"Falha ao salvar log em `{log_path}`: {e}")
