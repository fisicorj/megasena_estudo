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

# PDF (reportlab)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.pdfgen import canvas
from reportlab.lib import colors


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

def is_cloud() -> bool:
    return bool(os.getenv("STREAMLIT_CLOUD") or os.getenv("STREAMLIT_RUNTIME") or os.getenv("STREAMLIT_SERVER_HEADLESS"))

def normalize_weights(wf, wr):
    if abs((wf + wr) - 1.0) > 1e-9:
        s = wf + wr
        if s <= 0:
            return 0.5, 0.5
        wf = wf / s
        wr = 1.0 - wf
    return wf, wr

def make_key(seed_run, excel_sha, params: dict):
    base = json.dumps({"seed": seed_run, "sha": excel_sha, "params": params}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(base).hexdigest()

def parse_seed_text(seed_text: str) -> int:
    """
    Aceita:
      - decimal: "12345"
      - hex: "0xDEADBEEF"
    Retorna uint64 (0..2^64-1), ou levanta ValueError.
    """
    s = (seed_text or "").strip().lower()
    if not s:
        raise ValueError("Seed vazia.")

    base = 10
    if s.startswith("0x"):
        base = 16

    try:
        val = int(s, base=base)
    except Exception:
        raise ValueError("Seed inválida. Use decimal (ex: 123) ou hex (ex: 0xDEADBEEF).")

    if val < 0 or val > (2**64 - 1):
        raise ValueError("Seed fora do intervalo uint64 (0..2^64-1).")

    return val


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
    df = pd.read_excel(BytesIO(data), engine="openpyxl")
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
    step = max(1, hard_limit // 220)

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
# METRICS / SUMMARY
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

def build_exec_summary(result: dict) -> list[str]:
    params = result["params"]
    m20 = result["coverage_metrics_20"]
    mt = result["coverage_metrics_total"]

    bullets = []
    bullets.append(f"Execução em **{result['timestamp_sp']}** com seed **{result['seed']}** (reprodutível).")
    bullets.append(f"Base: **{result['rows']}** sorteios válidos • Arquivo **{result['excel_name']}** (SHA256 `{result['excel_sha256'][:12]}…`).")
    bullets.append(f"Pool válido: **{result['pool_len']:,}** jogos (tentativas: {result['pool_tries']:,}).")
    bullets.append(f"Cobertura: **{len(result['games_coverage'])}** jogos • **{m20['distinct']}** números distintos • peak = **{m20['peak']}**.")

    if params["n_top"] > 0:
        bullets.append(f"TOP: **{len(result['games_top'])}** jogos adicionais (overlap ≤ {params['top_rules']['max_overlap']}, máx/num ≤ {params['top_rules']['max_count_total']}).")
    else:
        bullets.append("TOP: desativado (N_TOP = 0).")

    bullets.append(f"Config: pesos freq={params['w_freq']:.2f} / recência={params['w_recency']:.2f} (λ={params['recency_lambda']:.3f}) • β={params['cover_beta']:.2f}.")
    bullets.append(f"Após TOP: distintos = **{mt['distinct']}** • peak total = **{mt['peak']}**.")

    if m20["most_common"]:
        top_used = ", ".join([f"{x['n']:02d}×{x['count']}" for x in m20["most_common"][:5]])
        bullets.append(f"Números mais usados (20 jogos): {top_used}.")

    return bullets

def build_report_text(result: dict) -> str:
    lines = []
    lines.append("RELATÓRIO — GERAÇÃO DE JOGOS (MEGA-SENA)")
    lines.append("=" * 48)
    lines.append(f"Data/Hora (SP): {result['timestamp_sp']}")
    lines.append(f"Seed: {result['seed']}")
    lines.append(f"Arquivo: {result['excel_name']} | SHA256: {result['excel_sha256']}")
    lines.append(f"Sorteios válidos: {result['rows']}")
    lines.append("")
    lines.append("Resumo executivo:")
    for b in build_exec_summary(result):
        lines.append(f"- {b}")
    lines.append("")
    lines.append("Jogos com cobertura:")
    for i, g in enumerate(result["games_coverage"], 1):
        lines.append(f"{i:02d}: {fmt_nums(g)}")
    lines.append("")
    lines.append("TOP (sem estragar cobertura):")
    if result["games_top"]:
        for i, t in enumerate(result["games_top"], 1):
            lines.append(f"TOP {i}: {fmt_nums(t['nums'])} | score={t['score']:.4f} | overlap={t['overlap_with_20']}/6")
    else:
        lines.append("[Sem TOP adicional]")
    lines.append("")
    lines.append("Métricas (20 jogos):")
    lines.append(json.dumps(result["coverage_metrics_20"], ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("Métricas (total):")
    lines.append(json.dumps(result["coverage_metrics_total"], ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("Parâmetros:")
    lines.append(json.dumps(result["params"], ensure_ascii=False, indent=2))
    return "\n".join(lines)


# =========================
# PDF REPORT (reportlab) - Relatório textual
# =========================
def build_report_pdf(result: dict) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Relatório Mega-Sena",
        author="Gerador Mega-Sena",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", fontSize=16, leading=20, spaceAfter=14))
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="Mono", fontName="Courier", fontSize=9, leading=12))

    story = []
    story.append(Paragraph("Relatório – Geração de Jogos (Mega-Sena)", styles["TitleBig"]))
    story.append(Paragraph(f"Data/Hora (SP): {result['timestamp_sp']}", styles["Small"]))
    story.append(Paragraph(f"Seed: {result['seed']}", styles["Small"]))
    story.append(Paragraph(f"Arquivo: {result['excel_name']}<br/>SHA256: {result['excel_sha256']}", styles["Small"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Resumo executivo</b>", styles["Heading2"]))
    bullets = build_exec_summary(result)
    story.append(
        ListFlowable(
            [ListItem(Paragraph(b, styles["Normal"])) for b in bullets],
            bulletType="bullet",
            start="circle",
            leftIndent=18,
        )
    )
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Jogos com cobertura</b>", styles["Heading2"]))
    for i, g in enumerate(result["games_coverage"], 1):
        story.append(Paragraph(f"{i:02d}: {fmt_nums(g)}", styles["Mono"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Jogos TOP</b>", styles["Heading2"]))
    if result["games_top"]:
        for i, t in enumerate(result["games_top"], 1):
            story.append(
                Paragraph(
                    f"TOP {i}: {fmt_nums(t['nums'])} "
                    f"(score={t['score']:.4f}, overlap={t['overlap_with_20']}/6)",
                    styles["Mono"],
                )
            )
    else:
        story.append(Paragraph("Nenhum jogo TOP adicional respeitou as restrições.", styles["Normal"]))

    story.append(Spacer(1, 14))
    story.append(Paragraph("<b>Métricas (20 jogos)</b>", styles["Heading2"]))
    story.append(Paragraph(json.dumps(result["coverage_metrics_20"], ensure_ascii=False, indent=2), styles["Mono"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Métricas (total)</b>", styles["Heading2"]))
    story.append(Paragraph(json.dumps(result["coverage_metrics_total"], ensure_ascii=False, indent=2), styles["Mono"]))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# =========================
# PDF CANHOTO - VISUAL "APP" (somente números do jogo)
# 2 jogos por página
# TOP 2 separado em nova página (2 jogos por página)
# =========================
def build_canhoto_pdf_visual_only_picks(
    result: dict,
    concurso: str = "",
    sorteio: str = "",
    titulo: str = "mega-sena",
) -> bytes:
    """
    PDF estilo 'volante/app' (não oficial), inspirado no visual:
      - Barra superior verde
      - Em vez de 01..60, mostra SOMENTE as dezenas sugeridas em "bolinhas"
      - 2 jogos por página
      - TOP 2 separado em nova página
    """

    # Cores aproximadas
    GREEN = colors.HexColor("#0B8F3B")
    GREEN_DARK = colors.HexColor("#087233")
    GREY_STROKE = colors.HexColor("#C8CDD0")
    WHITE = colors.white
    BG = colors.HexColor("#F7FAF8")

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    margin = 12 * mm
    gap_block = 10 * mm

    # 2 blocos por página (um em cima, um embaixo)
    block_h = (H - 2 * margin - gap_block) / 2
    block_w = W - 2 * margin

    header_h = 16 * mm
    inner_pad = 10 * mm

    def draw_top_bar(y_top, concurso_txt, sorteio_txt):
        """Barra verde no topo do bloco."""
        c.setFillColor(GREEN)
        c.setStrokeColor(GREEN)
        c.rect(margin, y_top - header_h, block_w, header_h, stroke=0, fill=1)

        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(margin + inner_pad, y_top - header_h + 5.2 * mm, titulo)

        c.setFont("Helvetica", 10)
        conc = concurso_txt.strip() or "Concurso ____"
        sort = sorteio_txt.strip() or "Sorteio __/__/__"
        right_text = f"{conc}   •   {sort}"
        tw = c.stringWidth(right_text, "Helvetica", 10)
        c.drawString(margin + block_w - inner_pad - tw, y_top - header_h + 5.8 * mm, right_text)

    def draw_block_frame(y_bottom):
        c.setFillColor(BG)
        c.setStrokeColor(colors.HexColor("#E7ECEE"))
        c.setLineWidth(1)
        c.roundRect(margin, y_bottom, block_w, block_h, 10, stroke=1, fill=1)

    def draw_game_label(x, y, label, idx, kind):
        c.setFillColor(GREEN_DARK if kind == "TOP" else GREEN)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, f"{idx:02d} • {label}")

    def draw_picks_balls(x0, y0, w, h, nums):
        """
        Desenha apenas as dezenas do jogo (6 bolinhas),
        com estilo tipo "selecionadas".
        """
        nums = sorted([int(n) for n in nums])

        # círculo grande, espaçado
        r = min(w / 12, h / 3)  # raio
        gap = r * 0.9

        total_w = 6 * (2 * r) + 5 * gap
        start_x = x0 + (w - total_w) / 2 + r
        cy = y0 + h / 2

        c.setFont("Helvetica-Bold", 16)
        for i, n in enumerate(nums):
            cx = start_x + i * (2 * r + gap)

            c.setFillColor(GREEN)
            c.setStrokeColor(GREEN)
            c.setLineWidth(1.2)
            c.circle(cx, cy, r, stroke=1, fill=1)

            c.setFillColor(WHITE)
            c.drawCentredString(cx, cy - 5.5, f"{n:02d}")

        # linha “assinatura” no rodapé do bloco
        c.setStrokeColor(colors.HexColor("#D1D9DD"))
        c.setLineWidth(0.8)
        c.line(x0, y0 + 6 * mm, x0 + w, y0 + 6 * mm)
        c.setFillColor(colors.HexColor("#6B7A7E"))
        c.setFont("Helvetica", 8)
        c.drawString(x0, y0 + 2.5 * mm, "Conferido / Assinatura")

    def draw_block(y_bottom, kind, idx_global, nums, label):
        y_top = y_bottom + block_h
        draw_block_frame(y_bottom)
        draw_top_bar(y_top, concurso, sorteio)

        label_y = y_top - header_h - 7 * mm
        draw_game_label(margin + inner_pad, label_y, label, idx_global, kind)

        # área das bolinhas
        balls_top = label_y - 10 * mm
        balls_bottom = y_bottom + 12 * mm
        balls_h = max(10 * mm, balls_top - balls_bottom)
        balls_w = block_w - 2 * inner_pad

        draw_picks_balls(margin + inner_pad, balls_bottom, balls_w, balls_h, nums)

    # ---------- Listas ----------
    cobertura = [(i + 1, "COBERTURA", g) for i, g in enumerate(result.get("games_coverage", []))]
    top2 = []
    if result.get("games_top"):
        for i, t in enumerate(result["games_top"], 1):
            top2.append((i, f"TOP {i}", t["nums"]))

    def render_list(items, kind, page_title=None):
        per_page = 2
        for i, (idx_local, label, nums) in enumerate(items, start=1):
            pos = (i - 1) % per_page
            if pos == 0:
                if i != 1:
                    c.showPage()

                if page_title:
                    c.setFillColor(GREEN_DARK if kind == "TOP" else GREEN)
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(margin, H - margin + 2 * mm, page_title)
                    c.setStrokeColor(colors.HexColor("#D8DEE2"))
                    c.setLineWidth(1)
                    c.line(margin, H - margin, W - margin, H - margin)

            # bloco de cima ou baixo
            y_bottom = margin + block_h + gap_block if pos == 0 else margin

            # índice global: cobertura usa sequência 01..N; top usa 01..2
            idx_global = i if kind != "TOP" else idx_local

            draw_block(
                y_bottom=y_bottom,
                kind=kind,
                idx_global=idx_global,
                nums=nums,
                label=label,
            )

    # Cobertura
    render_list(cobertura, kind="COB", page_title=None)

    # TOP separado em nova página
    if top2:
        c.showPage()
        render_list(top2, kind="TOP", page_title="TOP 2 (separado)")

    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf


# =========================
# UI (visual)
# =========================
def inject_css(compact: bool):
    base = """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; }
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] { padding: 10px 14px; border-radius: 12px; }
      .muted { opacity: .75; }
      .card {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 14px 16px;
      }
      .card-title { font-size: 0.9rem; opacity: .85; margin-bottom: 6px; }
      .card-value { font-size: 1.35rem; font-weight: 750; line-height: 1.2; }
      .card-sub { font-size: 0.85rem; opacity: .78; margin-top: 6px; }
      .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
      @media (max-width: 1100px) { .grid { grid-template-columns: repeat(2, 1fr); } }
      @media (max-width: 600px) { .grid { grid-template-columns: repeat(1, 1fr); } }
      code { border-radius: 8px; padding: 2px 6px; }
      .hr { height: 1px; background: rgba(255,255,255,0.08); margin: 12px 0; }
    </style>
    """
    compact_css = """
    <style>
      .block-container { max-width: 980px; }
      header, footer { visibility: hidden; height: 0; }
      .stSidebar { display: none; }
      .stTabs [data-baseweb="tab-list"] { display: none; }
      .card { padding: 10px 12px; border-radius: 14px; }
      .card-value { font-size: 1.15rem; }
    </style>
    """ if compact else ""
    st.markdown(base + compact_css, unsafe_allow_html=True)

def icon_card(icon: str, title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{icon} {title}</div>
          <div class="card-value">{value}</div>
          {f'<div class="card-sub">{sub}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

def style_top_table(df: pd.DataFrame):
    return df


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Mega-Sena • Cobertura + Score", layout="wide")

if "compact" not in st.session_state:
    st.session_state["compact"] = False

# Sidebar
with st.sidebar:
    st.header("📥 Entrada")
    up = st.file_uploader("Upload do Excel (.xlsx)", type=["xlsx"])

    st.divider()
    st.header("🧾 Exibição")
    compact_toggle = st.toggle(
        "Modo compact (impressão/relatório)",
        value=st.session_state["compact"],
        key="compact_toggle_sidebar",
    )
    st.session_state["compact"] = compact_toggle

    modo_didatico = st.toggle("Modo didático (avisos)", value=False)

    st.divider()
    st.header("⚙️ Execução")
    run_btn = st.button("🚀 Gerar jogos", type="primary", use_container_width=True)

    seed_mode = st.radio("Seed", ["Aleatória", "Fixar"], horizontal=True)

    seed_text = None
    seed_error = None
    if seed_mode == "Fixar":
        seed_text = st.text_input(
            "Seed fixa (uint64)",
            value=st.session_state.get("seed_text", "123"),
            help="Use decimal (ex: 123) ou hex (ex: 0xDEADBEEF). Intervalo: 0..2^64-1.",
        )
        st.session_state["seed_text"] = seed_text
        try:
            _ = parse_seed_text(seed_text)
        except Exception as e:
            seed_error = str(e)
            st.error(seed_error)

    st.divider()
    st.header("🧠 Parâmetros")
    pool_size = st.number_input("POOL_SIZE", min_value=10_000, max_value=2_000_000, value=200_000, step=10_000)
    scan_top = st.number_input("SCAN_TOP (cobertura)", min_value=5_000, max_value=300_000, value=50_000, step=5_000)
    n_cobertura = st.number_input("N_COBERTURA (jogos)", min_value=5, max_value=50, value=20, step=1)
    n_top = st.number_input("N_TOP (top jogos)", min_value=0, max_value=10, value=2, step=1)

    st.divider()
    st.header("📌 Pesos")
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
    st.header("🛡️ Cobertura/TOP")
    cover_beta = st.slider("COVER_BETA", 0.0, 2.0, 0.45, 0.01)
    max_count_total = st.number_input("MAX_COUNT_PER_NUMBER_TOTAL", 1, 10, 3, 1)
    max_overlap = st.number_input("MAX_OVERLAP_WITH_COVERAGE", 0, 6, 4, 1)
    dont_increase_peak = st.toggle("DONT_INCREASE_PEAK", value=True)

    st.divider()
    st.header("🧾 Log")
    enable_log = st.toggle("Salvar log JSONL", value=False)
    log_path = st.text_input("LOG_PATH", value=DEFAULT_LOG_PATH, disabled=not enable_log)

# Apply CSS
modo_compact = bool(st.session_state["compact"])
inject_css(modo_compact)

def exit_compact():
    st.session_state["compact"] = False
    st.session_state["compact_toggle_sidebar"] = False
    st.rerun()

if modo_compact:
    cols = st.columns([2, 6, 2])
    with cols[0]:
        st.button("⬅️ Voltar do modo compact", use_container_width=True, on_click=exit_compact)

st.markdown("## 🎲 Mega-Sena — Gerador (cobertura + top score)")
st.markdown('<div class="muted">Gere combinações com seed registrada, cobertura de números e seleção por score.</div>', unsafe_allow_html=True)

if modo_didatico:
    st.info(
        "📌 **Observação didática**: o algoritmo não prevê resultados. "
        "Ele usa frequência/recência como ponderação para estudo estatístico e aplica heurísticas + cobertura."
    )

# Load dataset
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

top_pos = top_by_position_with_pct(df, topk=10)
top_all = top_overall(df, topk=15)

if "runs" not in st.session_state:
    st.session_state["runs"] = {}
if "last_run_key" not in st.session_state:
    st.session_state["last_run_key"] = None

tab_stats, tab_run, tab_report, tab_export, tab_debug = st.tabs(
    ["📊 Estatísticas", "✅ Jogos", "🧾 Relatório", "⬇️ Exportar", "🧪 Debug"]
)

# =========================
# Estatísticas
# =========================
with tab_stats:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top por posição")
        for col in BOLAS_COLS:
            tdf = pd.DataFrame(top_pos[col])
            st.markdown(f"**{col}**")
            st.dataframe(style_top_table(tdf), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top geral")
        tdf = pd.DataFrame(top_all)
        st.dataframe(style_top_table(tdf), use_container_width=True, hide_index=True)

        chart_df = tdf.copy()
        chart_df["n"] = chart_df["n"].map(lambda x: f"{int(x):02d}")
        chart_df = chart_df.set_index("n")[["count"]]
        st.caption("Top 15 por frequência (barra):")
        st.bar_chart(chart_df)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Execução
# =========================
with tab_run:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Geração de jogos")
    st.caption("A geração pesada só roda quando você clica em **Gerar jogos** na sidebar.")

    if run_btn:
        if seed_mode == "Fixar" and seed_error:
            st.error("Corrija a seed fixa antes de gerar.")
            st.stop()

        if seed_mode == "Fixar":
            seed_run = parse_seed_text(seed_text)
        else:
            seed_run = secrets.randbits(64)

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

        if run_key in st.session_state["runs"]:
            result = st.session_state["runs"][run_key]
            st.success(f"✅ Reutilizado do cache da sessão • seed={seed_run}")
        else:
            st.success(f"Iniciando: {now_sp()} • seed={seed_run}")

            with st.spinner("Calculando pesos..."):
                p = build_weights(df, w_freq=w_freq_n, w_recency=w_recency_n, recency_lambda=float(recency_lambda))

            prog = st.progress(0.0)
            status = st.empty()

            def progress_cb(tries, hard_limit, pool_len):
                frac = min(tries / max(1, hard_limit), 1.0)
                prog.progress(frac)
                status.write(f"Tentativas: **{tries:,}** / {hard_limit:,} • Pool válido: **{pool_len:,}**")

            with st.spinner("Gerando pool com heurísticas..."):
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
                "seed": int(seed_run),
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
                "run_key": run_key,
            }

            st.session_state["runs"][run_key] = result

            if enable_log:
                if is_cloud():
                    st.warning("Log desativado automaticamente em ambiente cloud/público.")
                else:
                    try:
                        save_log(result, log_path)
                        st.info(f"Log salvo em: `{log_path}`")
                    except Exception as e:
                        st.warning(f"Falha ao salvar log: {e}")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="grid">', unsafe_allow_html=True)
        icon_card("🧬", "Seed", str(result["seed"]), "Reproduzível")
        icon_card("🔑", "Run key", result["run_key"][:10] + "…", "Identificador")
        icon_card("🧺", "Pool válido", f"{result['pool_len']:,}", f"Tentativas: {result['pool_tries']:,}")
        icon_card("🧩", "Distintos (20)", str(result["coverage_metrics_20"]["distinct"]), f"Peak: {result['coverage_metrics_20']['peak']}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        st.markdown("### ✅ Jogos com cobertura")
        jogos_df = pd.DataFrame(result["games_coverage"], columns=[f"N{i}" for i in range(1, 7)])
        st.dataframe(jogos_df, use_container_width=True, hide_index=True)

        st.markdown("### 🏆 TOP (sem estragar a cobertura)")
        if len(result["games_top"]) == 0 and result["params"]["n_top"] > 0:
            st.warning("Não encontrei TOPs que respeitem as restrições. Afrouxe os limites.")
        else:
            top_df = pd.DataFrame(
                [
                    {"Jogo": fmt_nums(t["nums"]), "Score": round(float(t["score"]), 4), "Overlap com 20": f"{int(t['overlap_with_20'])}/6"}
                    for t in result["games_top"]
                ]
            )
            st.dataframe(top_df, use_container_width=True, hide_index=True)

        # ✅ Canhoto também aqui (aparece sempre após gerar)
        st.markdown("### 🎟️ Canhoto (visual mega-sena — apenas dezenas sugeridas)")
        canhoto_pdf = build_canhoto_pdf_visual_only_picks(result)
        st.download_button(
            "🎟️ Baixar canhoto (PDF)",
            data=canhoto_pdf,
            file_name=f"canhoto_visual_megasena_{result['seed']}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"dl_canhoto_run_{result['seed']}",
        )

    else:
        st.info("Clique em **Gerar jogos** para rodar. Resultados ficam em cache na sessão.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Relatório
# =========================
with tab_report:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Resumo executivo + Relatório + PDFs")

    last_key = st.session_state.get("last_run_key")
    runs = st.session_state.get("runs", {})
    if not last_key or last_key not in runs:
        st.info("Rode uma geração na aba **Jogos** para criar o relatório.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    result = runs[last_key]
    bullets = build_exec_summary(result)

    st.markdown("### 🧾 Resumo executivo (automático)")
    for b in bullets:
        st.markdown(f"- {b}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### 🎟️ Canhoto (estilo app) — PDF")
    canhoto_pdf = build_canhoto_pdf_visual_only_picks(result)
    st.download_button(
        "🎟️ Baixar canhoto (PDF estilo app)",
        data=canhoto_pdf,
        file_name=f"canhoto_visual_megasena_{result['seed']}.pdf",
        mime="application/pdf",
        use_container_width=True,
        key=f"dl_canhoto_report_{result['seed']}",
    )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### 🧾 Relatório (copiar/colar)")
    report_text = build_report_text(result)
    st.text_area("Relatório pronto", report_text, height=420)

    st.download_button(
        "Baixar relatório (TXT)",
        data=report_text.encode("utf-8"),
        file_name=f"relatorio_megasena_{result['seed']}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    pdf_bytes = build_report_pdf(result)
    st.download_button(
        "📄 Baixar relatório em PDF (texto)",
        data=pdf_bytes,
        file_name=f"relatorio_megasena_{result['seed']}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.caption("Para imprimir: ative **Modo compact** e use Ctrl+P no navegador.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Export
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

    top_df = pd.DataFrame(
        [
            {"Jogo": fmt_nums(t["nums"]), "Score": round(float(t["score"]), 4), "Overlap com 20": f"{int(t['overlap_with_20'])}/6"}
            for t in result["games_top"]
        ]
    ) if len(result["games_top"]) else pd.DataFrame(columns=["Jogo", "Score", "Overlap com 20"])

    st.download_button(
        "Baixar JSON do resultado",
        data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"megasena_result_{result['seed']}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.download_button(
        "Baixar CSV (20 jogos)",
        data=jogos_df.to_csv(index=False).encode("utf-8"),
        file_name=f"megasena_20_{result['seed']}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        jogos_df.to_excel(writer, sheet_name="20_Jogos", index=False)
        top_df.to_excel(writer, sheet_name="Top", index=False)
        pd.DataFrame(result["top_overall"]).to_excel(writer, sheet_name="Top_Geral", index=False)

        rows = []
        for col in BOLAS_COLS:
            for item in result["top_position"][col]:
                rows.append({"posicao": col, **item})
        pd.DataFrame(rows).to_excel(writer, sheet_name="Top_Posicao", index=False)

        pd.DataFrame([result["coverage_metrics_20"]]).to_excel(writer, sheet_name="Metricas_20", index=False)
        pd.DataFrame([result["coverage_metrics_total"]]).to_excel(writer, sheet_name="Metricas_Total", index=False)
        pd.json_normalize(result["params"]).to_excel(writer, sheet_name="Params", index=False)

        report_text = build_report_text(result)
        pd.DataFrame({"Relatorio": report_text.splitlines()}).to_excel(writer, sheet_name="Relatorio", index=False)

    st.download_button(
        "Baixar Excel completo (abas)",
        data=out.getvalue(),
        file_name=f"megasena_{result['seed']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Debug
# =========================
with tab_debug:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Debug / Diagnóstico")
    st.markdown(f"- Arquivo: **{excel_name}**")
    st.markdown(f"- SHA256: `{excel_sha}`")
    st.markdown(f"- Linhas válidas: **{len(df)}**")

    with st.expander("Prévia (15 linhas)", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("**Cache da sessão**")
    st.write(f"Runs armazenados: {len(st.session_state.get('runs', {}))}")
    st.markdown("</div>", unsafe_allow_html=True)
