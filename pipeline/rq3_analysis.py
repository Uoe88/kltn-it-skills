#!/usr/bin/env python3
"""
rq3_analysis.py  -  Câu hỏi nghiên cứu 3

"Độ rộng kỹ năng và độ phức tạp tech stack của từng nhóm nghề (Frontend, Backend, Data Engineer,
 Data Scientist, AI Engineer) thể hiện ra sao ở các mức level khác nhau, và các level cao (Senior, Lead+)
 có thêm những kỹ năng khó, hiếm mà level thấp không có hay không?"

THUẬT TOÁN
Bước 1: Xác định trục phân tích level: Entry, Middle, Senior, Lead+ (level_group)
Bước 2: Chỉ số độ rộng: số hard skill đã trừ nhãn lĩnh vực và loại trùng cha-con, cộng số nhóm skill (độ phức tạp tech stack).
Bước 3: Skill hiếm theo nghề: có tần suất dưới 5% trong JD của chính nghề đó, tính trên mọi level. (kèm danh sách cần duyệt)
Bước 4: Xu hướng theo level trong từng nghề: Spearman giữa level thứ bậc (0-3) và chỉ số, CI 95% bootstrap, hiệu chỉnh Holm giữa các nghề.
Bước 5: Bảng nghề x level: median, IQR, mean (chỉ số nhị phân thì báo tỷ lệ kèm CI Wilson).
Bước 6: Skill hiếm của nghề: số skill hiếm, số skill hiếm lộ ra khi lấy mẫu con cùng cỡ.
Bước 7: Skill theo level: so nhóm thấp (Entry + Middle) với nhóm cao (Senior + Lead+) bằng chênh điểm phần trăm và kiểm định Fisher (Holm trong nghề); cờ only_high đánh dấu skill chỉ có ở nhóm cao, so với đường cơ sở hoán vị nhãn level. 
BƯỚC 8: Output rq3_by_level.csv, rq3_rare_skills.csv, rq3_level_skills.csv. 

Chạy: python rq3_analysis.py --input jd_skill_binary.csv --outdir rq3_out [--taxonomy skill_taxonomy.csv --hard-skills hard_skills.csv --report]
"""
import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# <<<SHARED-START (khối dùng chung RQ1/RQ3, phải giống hệt ở hai file)
TARGET_ROLES = ["Backend", "Frontend", "Data Engineer", "Data Scientist", "AI Engineer"]
LANGUAGE_SKILL_COLS = {"skill_english", "skill_japanese", "skill_french"}

DEFAULT_DOMAIN_COLS = "artificial_intelligence,machine_learning,deep_learning,nlp,computer_vision"

DEFAULT_IMPLIED = [
    ("skill_spring_boot", "skill_spring"), ("skill_jenkins", "skill_ci_cd"), ("skill_github_actions", "skill_ci_cd"),
    ("skill_gitlab_ci", "skill_ci_cd"), ("skill_next_js", "skill_react"),
    ("skill_redshift", "skill_aws"), ("skill_redshift", "skill_data_warehouse"), ("skill_dynamodb", "skill_aws"),
    ("skill_dynamodb", "skill_nosql"), ("skill_bigquery", "skill_gcp"), ("skill_bigquery", "skill_data_warehouse"),
    ("skill_snowflake", "skill_data_warehouse"), ("skill_azure_synapse", "skill_azure"),
    ("skill_azure_synapse", "skill_data_warehouse"), ("skill_vertex_ai", "skill_gcp"), ("skill_sagemaker", "skill_aws"),
]


def wilson_ci(k, n, z=1.96):
    """Khoảng tin cậy Wilson 95% cho tỷ lệ k/n (trả về tỷ lệ 0-1)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else np.nan


def norm_skill(name):
    name = name.strip()
    return name if name.startswith("skill_") else f"skill_{name}"


def short(col):
    return col.replace("skill_", "")


def parse_domain_cols(text):
    return [norm_skill(c) for c in text.split(",") if c.strip()]


def slug_skill(name):
    return "skill_" + re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def load_group_map(taxonomy_path):
    """skill_<slug> -> group (từ skill_taxonomy.csv). Trả về {} nếu không có file."""
    if not taxonomy_path or not Path(taxonomy_path).exists():
        return {}
    t = pd.read_csv(taxonomy_path, encoding="utf-8-sig")
    if not {"canonical", "group"} <= set(t.columns):
        return {}
    return {slug_skill(c): g for c, g in zip(t.canonical, t.group)}


def load_data(path, domain_cols, group_map=None):
    """Đọc file, chọn cột công nghệ, điều chỉnh độ rộng cho nhãn lĩnh vực."""
    df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df = df.copy()
    skill_all = [c for c in df.columns if c.startswith("skill_") and c not in LANGUAGE_SKILL_COLS]
    unknown = [c for c in domain_cols if c not in skill_all]
    if unknown:
        raise SystemExit(f"--domain-cols có cột không tồn tại (hoặc là ngoại ngữ): {unknown}")
    skill_cols = [c for c in skill_all if c not in domain_cols]
    df[skill_all] = df[skill_all].fillna(0).astype(int)
    df["has_skills"] = df["has_skills"].fillna(0).astype(int)

    df["n_hard_skills_file"] = df["n_hard_skills"]
    n_dom = df[domain_cols].sum(axis=1) if domain_cols else 0
    df["n_hard_skills"] = (df["n_hard_skills_file"] - n_dom).clip(lower=0)
    df["n_hard_skills_in_cols"] = df[skill_cols].sum(axis=1)

    df["n_skill_groups_file"] = df["n_skill_groups"]
    df["groups_adjusted"] = False
    if domain_cols and group_map:
        drop_groups = {group_map[c] for c in domain_cols if c in group_map}
        lost = pd.Series(0, index=df.index)
        for grp in drop_groups:
            dom_in = [c for c in domain_cols if group_map.get(c) == grp]
            others = [c for c in skill_cols if group_map.get(c) == grp]
            has_dom = df[dom_in].sum(axis=1) > 0
            has_other = df[others].sum(axis=1) > 0 if others else pd.Series(False, index=df.index)
            lost += (has_dom & ~has_other).astype(int)
        df["n_skill_groups"] = (df["n_skill_groups_file"] - lost).clip(lower=0)
        df["groups_adjusted"] = lost > 0
    return df, skill_cols


def resolve_implied(arg, input_path, df, skill_cols):
    """Lấy danh sách (con, cha) của luật roll-up. Ưu tiên skill_implied_pairs.csv (build_skill_taxonomy.py xuất);"""
    path = Path(arg) if arg else Path(input_path).parent / "skill_implied_pairs.csv"
    cols = set(skill_cols)
    if path.exists():
        t = pd.read_csv(path, encoding="utf-8-sig")
        pairs = list(zip(t["child"], t["implied_parent"]))
        source = f"{path}"
    else:
        h = df[df.has_skills == 1]
        pairs = [(c, p) for c, p in DEFAULT_IMPLIED
                 if c in cols and p in cols and h[c].sum() > 0 and h.loc[h[c] == 1, p].mean() >= 0.99]
        source = "danh sách dựng sẵn, đã kiểm chứng P(cha|con) >= 99% (không thấy skill_implied_pairs.csv)"
    pairs = [(c, p) for c, p in pairs if c in cols and p in cols]
    return pairs, source


def add_dedup(df, implied):
    """n_hard_skills_dedup = n_hard_skills - số họ cha-con bị đếm trùng (cha chỉ trừ khi có >= 1 con cùng xuất hiện)."""
    by_parent = {}
    for c, p in implied:
        by_parent.setdefault(p, []).append(c)
    red = pd.Series(0, index=df.index)
    for p, cs in by_parent.items():
        red = red + ((df[p] == 1) & (df[cs].sum(axis=1) > 0)).astype(int)
    df["n_implied_redundant"] = red
    df["n_hard_skills_dedup"] = (df["n_hard_skills"] - red).clip(lower=0)
    return df


def filter_jobs(df, roles):
    """has_skills == 1 và (nếu có cột skills_status) chỉ giữ 'has_hard': loại JD chỉ có noise, vì chúng"""
    g = df[df.expertise_category.isin(roles) & (df.has_skills == 1)]
    n_noise = 0
    if "skills_status" in g.columns:
        n_noise = int((g.skills_status != "has_hard").sum())
        g = g[g.skills_status == "has_hard"]
    return g.copy(), n_noise
# >>>SHARED-END

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", stats.ConstantInputWarning) if hasattr(stats, "ConstantInputWarning") else None

LEVELS = ["Entry", "Middle", "Senior", "Lead+"]
LEVEL_ORD = {lv: i for i, lv in enumerate(LEVELS)}
LOW_LEVELS = ("Entry", "Middle")
HIGH_LEVELS = ("Senior", "Lead+")
ALL = "TẤT CẢ"

# Tiêu chí: hạ tầng / điều phối, xử lý phân tán và streaming, thiết kế hệ thống, MLOps/RAG; thường kỳ vọng ở kinh nghiệm cao.
DEFAULT_HARD_SKILLS = [
    "kubernetes", "helm", "argocd", "terraform", "ansible",
    "kafka", "flink", "spark", "hadoop", "scala", "airflow", "dbt", "databricks",
    "microservices", "event_driven", "software_architecture", "mlops", "rag",
]

METRICS = [("n_hard_skills_dedup", False),
           ("n_hard_skills", False),
           ("n_skill_groups", False),
           ("n_rare_skills", False), ("has_rare_skill", True),
           ("n_hard_manual", False), ("has_hard_manual", True),
           ("n_tail_skills", False)]


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1)
    return out


# BƯỚC 1  DANH SÁCH KỸ NĂNG KHÓ (THỦ CÔNG)
def resolve_hard_skills(arg, skill_cols):
    """Trả về (cột dùng được, tên không có cột, mô tả nguồn)."""
    cols = set(skill_cols)
    if arg:
        t = pd.read_csv(arg, encoding="utf-8-sig")
        names = t["skill"] if "skill" in t.columns else t.iloc[:, 0]
        wanted, source = [norm_skill(str(s)) for s in names], str(arg)
    else:
        wanted, source = [norm_skill(s) for s in DEFAULT_HARD_SKILLS], "đề xuất khởi đầu trong code (cần bạn duyệt)"
    return [c for c in wanted if c in cols], [short(c) for c in wanted if c not in cols], source


# BƯỚC 3  "HIẾM THEO NGHỀ" VÀ ĐẶC TRƯNG TỪNG JD
def define_rare(g, skill_cols, roles, rare_pct):
    """Skill hiếm của nghề: 0 < tần suất trong nghề < rare_pct %. Tính trên mọi JD của nghề (mọi level)."""
    out = {}
    for role in roles:
        prev = g.loc[g.expertise_category == role, skill_cols].mean()
        out[role] = [c for c in skill_cols if 0 < prev[c] < rare_pct / 100]
    return out


def add_jd_features(g, rare_by_role, hard_cols):
    g = g.copy()
    g["n_rare_skills"] = 0
    for role, rc in rare_by_role.items():
        if rc:
            m = g.expertise_category == role
            g.loc[m, "n_rare_skills"] = g.loc[m, rc].sum(axis=1)
    g["has_rare_skill"] = (g.n_rare_skills > 0).astype(int)
    g["n_hard_manual"] = g[hard_cols].sum(axis=1) if hard_cols else 0
    g["has_hard_manual"] = (g.n_hard_manual > 0).astype(int)
    g["n_tail_skills"] = (g.n_hard_skills - g.n_hard_skills_in_cols).clip(lower=0)
    return g


# BƯỚC 4  XU HƯỚNG THEO LEVEL
def trend_table(g_lv, roles, n_boot, seed, min_n=20):
    """Spearman giữa level thứ bậc (Entry=0 ... Lead+=3) và từng chỉ số, trong từng nghề."""
    rng = np.random.default_rng(seed)
    rows = []
    for role in roles:
        gr = g_lv[g_lv.expertise_category == role]
        x = gr.level_group.map(LEVEL_ORD).to_numpy(float)
        for metric, _ in METRICS:
            y = gr[metric].to_numpy(float)
            row = {"role": role, "metric": metric, "n": len(gr), "rho": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
            if len(gr) >= min_n and np.ptp(x) > 0 and np.ptp(y) > 0:
                rho, p = stats.spearmanr(x, y)
                boots = []
                for _ in range(n_boot):
                    idx = rng.integers(0, len(gr), len(gr))
                    xb, yb = x[idx], y[idx]
                    if np.ptp(xb) > 0 and np.ptp(yb) > 0:
                        boots.append(stats.spearmanr(xb, yb)[0])
                row.update(rho=rho, p=p)
                if len(boots) >= 20:
                    row["ci_low"], row["ci_high"] = np.percentile(boots, [2.5, 97.5])
            rows.append(row)
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for metric, sub in t.groupby("metric"):            # Holm giữa các nghề trong cùng một chỉ số
        ok = sub.p.notna()
        if ok.any():
            t.loc[sub.index[ok], "p_holm"] = holm(sub.p[ok].to_numpy())
    return t


# BƯỚC 5  BẢNG NGHỀ x LEVEL
BY_LEVEL_COLS = ["role", "level", "metric", "n_jobs", "cell_n_small", "exp_min_mean", "n_with_exp",
                 "mean", "median", "q1", "q3", "ci_low", "ci_high",
                 "trend_n", "trend_rho", "trend_ci_low", "trend_ci_high", "trend_p_holm"]


def by_level_table(g_lv, roles, trend, min_cell):
    tr = trend.set_index(["role", "metric"])
    rows = []
    for role in roles:
        gr = g_lv[g_lv.expertise_category == role]
        for level in [ALL] + LEVELS:
            c = gr if level == ALL else gr[gr.level_group == level]
            if c.empty:
                continue
            exp = c.exp_min.dropna()
            for metric, is_bin in METRICS:
                x = c[metric]
                row = {"role": role, "level": level, "metric": metric, "n_jobs": len(c), "cell_n_small": len(c) < min_cell,
                       "exp_min_mean": round(exp.mean(), 2) if len(exp) else np.nan, "n_with_exp": len(exp),
                       "mean": round(x.mean(), 3), "median": np.nan, "q1": np.nan, "q3": np.nan, "ci_low": np.nan, "ci_high": np.nan}
                if is_bin:                                       # mean = tỷ lệ JD, CI Wilson
                    lo, hi = wilson_ci(int(x.sum()), len(x))
                    row["ci_low"], row["ci_high"] = round(lo, 3), round(hi, 3)
                else:
                    row.update(median=q(x, 50), q1=q(x, 25), q3=q(x, 75))
                if level == ALL and (role, metric) in tr.index:
                    t = tr.loc[(role, metric)]
                    row.update(trend_n=int(t.n), trend_rho=round(t.rho, 3), trend_ci_low=round(t.ci_low, 3),
                               trend_ci_high=round(t.ci_high, 3), trend_p_holm=round(t.p_holm, 4))
                rows.append(row)
    return pd.DataFrame(rows).reindex(columns=BY_LEVEL_COLS)


# BƯỚC 6  KỸ NĂNG HIẾM CỦA TỪNG NGHỀ
def rare_summary(g, skill_cols, roles, rare_by_role, hard_cols, rarefy_n, n_rep, rare_pct, seed):
    rng = np.random.default_rng(seed + 1)
    rows = []
    for role in roles:
        gr = g[g.expertise_category == role]
        n = len(gr)
        if n == 0:
            continue
        cnt = gr[skill_cols].sum()
        rc = rare_by_role[role]
        m = min(rarefy_n, n)
        distinct = []
        if rc:
            Xr = gr[rc].to_numpy(int)
            for _ in range(n_rep):                               # rarefaction: số skill hiếm lộ ra khi chỉ có m JD
                idx = rng.choice(n, size=m, replace=False)
                distinct.append(int((Xr[idx].sum(axis=0) > 0).sum()))
        mentions = int(cnt.sum())
        rows.append({
            "role": role, "n_jobs": n, "rare_pct_threshold": rare_pct,
            "n_skills_present": int((cnt > 0).sum()), "n_rare_skills": len(rc),
            "pct_rare_of_present": round(100 * len(rc) / max((cnt > 0).sum(), 1), 1),
            "rarefy_n_used": m,
            "n_rare_rarefied_mean": round(float(np.mean(distinct)), 1) if distinct else 0.0,
            "n_rare_rarefied_ci_low": round(float(np.percentile(distinct, 2.5)), 1) if distinct else 0.0,
            "n_rare_rarefied_ci_high": round(float(np.percentile(distinct, 97.5)), 1) if distinct else 0.0,
            "share_mentions_rare_pct": round(100 * cnt[rc].sum() / mentions, 1) if mentions and rc else 0.0,
            "mean_n_tail_skills": round(gr.n_tail_skills.mean(), 2),
            "n_hard_manual_present": int((cnt[hard_cols] > 0).sum()) if hard_cols else 0,
            "rare_skills": ", ".join(f"{short(c)}({int(cnt[c])})" for c in sorted(rc, key=lambda c: -cnt[c])),
        })
    return pd.DataFrame(rows)


# BƯỚC 7  KỸ NĂNG THEO LEVEL (NHÓM THẤP vs NHÓM CAO)
LEVEL_SKILL_COLS = ["role", "skill", "n_total", "pct_Entry", "pct_Middle", "pct_Senior", "pct_Lead+",
                    "k_low", "n_low", "pct_low", "k_high", "n_high", "pct_high", "diff_pp", "fisher_p", "p_holm_in_role",
                    "only_high", "is_rare_in_role", "is_hard_manual",
                    "role_only_high_obs", "role_only_high_perm_mean", "role_only_high_perm_p95"]


def level_skills_table(g_lv, skill_cols, roles, rare_by_role, hard_cols, min_total, min_high, min_group_n, n_perm, seed):
    rng = np.random.default_rng(seed + 2)
    rows, skipped, perm_info = [], [], {}
    for role in roles:
        gr = g_lv[g_lv.expertise_category == role]
        is_high = gr.level_group.isin(HIGH_LEVELS).to_numpy()
        n_high, n_low = int(is_high.sum()), int((~is_high).sum())
        if min(n_high, n_low) < min_group_n:
            skipped.append(f"{role} (thấp={n_low}, cao={n_high})")
            continue
        X = gr[skill_cols].to_numpy(int)
        tot = X.sum(axis=0)
        k_high = X[is_high].sum(axis=0)
        k_low = tot - k_high
        keep = tot >= min_total
        only_high = (k_low == 0) & (k_high >= min_high) & keep
        perm = []
        for _ in range(n_perm):                                   # đường cơ sở: xáo trộn nhãn nhóm thấp/cao
            kh = X[rng.choice(len(gr), size=n_high, replace=False)].sum(axis=0)
            perm.append(int((((tot - kh) == 0) & (kh >= min_high) & keep).sum()))
        obs, pmean, p95 = int(only_high.sum()), float(np.mean(perm)), float(np.percentile(perm, 95))
        perm_info[role] = (obs, pmean, p95, n_low, n_high)
        role_rows = []
        pct_lv = {lv: gr.loc[gr.level_group == lv, skill_cols].mean() * 100 for lv in LEVELS if (gr.level_group == lv).any()}
        rare = set(rare_by_role[role])
        for i, c in enumerate(skill_cols):
            if not keep[i]:
                continue
            _, p = stats.fisher_exact([[int(k_high[i]), n_high - int(k_high[i])], [int(k_low[i]), n_low - int(k_low[i])]])
            role_rows.append({
                "role": role, "skill": short(c), "n_total": int(tot[i]),
                **{f"pct_{lv}": round(pct_lv[lv][c], 1) if lv in pct_lv else np.nan for lv in LEVELS},
                "k_low": int(k_low[i]), "n_low": n_low, "pct_low": round(100 * k_low[i] / n_low, 1),
                "k_high": int(k_high[i]), "n_high": n_high, "pct_high": round(100 * k_high[i] / n_high, 1),
                "diff_pp": round(100 * k_high[i] / n_high - 100 * k_low[i] / n_low, 1), "fisher_p": p,
                "only_high": bool(only_high[i]), "is_rare_in_role": c in rare, "is_hard_manual": c in hard_cols,
                "role_only_high_obs": obs, "role_only_high_perm_mean": round(pmean, 1), "role_only_high_perm_p95": round(p95, 1)})
        if role_rows:
            ph = holm([r["fisher_p"] for r in role_rows])
            for r, pa in zip(role_rows, ph):
                r["p_holm_in_role"] = round(pa, 4)
                r["fisher_p"] = round(r["fisher_p"], 4)
            rows.extend(sorted(role_rows, key=lambda r: (not r["only_high"], -r["diff_pp"])))
    return pd.DataFrame(rows).reindex(columns=LEVEL_SKILL_COLS), skipped, perm_info


# BƯỚC 8  BÁO CÁO VÀ XUẤT FILE
def build_report(ctx):
    a, roles, by_level, trend, rare, lvl, perm_info = (ctx["a"], ctx["roles"], ctx["by_level"], ctx["trend"],
                                                       ctx["rare"], ctx["lvl"], ctx["perm_info"])
    rep = [f"Vị trí: {', '.join(roles)} | trục level: level_group ({' / '.join(LEVELS)}) | JD dùng: {ctx['n_jobs']}",
           f"Nhãn lĩnh vực đã loại: {', '.join(short(c) for c in ctx['domain_cols']) or '(không)'}; "
           f"n_skill_groups điều chỉnh ở {ctx['n_groups_adjusted']} JD" + ("" if ctx["has_group_map"] else " (KHÔNG có taxonomy nên chưa điều chỉnh)"),
           f"Lọc: has_hard; loại {ctx['n_noise']} JD chỉ có noise và {ctx['n_no_level']} JD thiếu level_group",
           f"Skill hiếm theo nghề: tần suất < {a.rare_pct:g}% trong JD của nghề (mọi level)",
           f"Kỹ năng khó thủ công ({ctx['hard_src']}): {', '.join(short(c) for c in ctx['hard_cols'])}"
           + (f" | không có cột/đã bị loại: {', '.join(ctx['hard_missing'])}" if ctx["hard_missing"] else ""), ""]

    rep.append("=== Kiểm tra trục level: exp_min trung bình theo level_group (gộp 5 nghề) ===")
    for lv, r in ctx["exp_by_level"].iterrows():
        rep.append(f"  {lv:<8} n_with_exp={int(r['count']):<4} mean={r['mean']:.2f}  median={r['median']:.1f}")

    rep.append("\n=== Độ rộng (n_hard_skills_dedup): median [Q1-Q3] theo nghề x level; [n] = số JD ===")
    for role in roles:
        parts = []
        for lv in [ALL] + LEVELS:
            r = by_level[(by_level.role == role) & (by_level.level == lv) & (by_level.metric == "n_hard_skills_dedup")]
            if r.empty:
                continue
            r = r.iloc[0]
            parts.append(f"{lv}: {r['median']:g} [{r.q1:g}-{r.q3:g}] [{r.n_jobs}]" + ("*" if r.cell_n_small else ""))
        rep.append(f"  {role:<15} " + " | ".join(parts))
    rep.append("  (* = ô có ít JD, chỉ mang tính tham khảo)")

    rep.append("\n=== Xu hướng theo level (Spearman, level thứ bậc 0-3; CI bootstrap 95%; Holm giữa các nghề) ===")
    for metric, _ in METRICS:
        rep.append(f"  [{metric}]")
        for r in trend[trend.metric == metric].itertuples():
            if np.isnan(r.rho):
                rep.append(f"    {r.role:<15} không đủ dữ liệu")
            else:
                rep.append(f"    {r.role:<15} n={r.n:<4} rho={r.rho:+.2f} [{r.ci_low:+.2f}; {r.ci_high:+.2f}]  p_holm={r.p_holm:.3f}")

    rep.append("\n=== Kỹ năng hiếm của từng nghề ===")
    for r in rare.itertuples():
        rep.append(f"  {r.role:<15} n={r.n_jobs:<4} skill hiếm: {r.n_rare_skills}/{r.n_skills_present} ({r.pct_rare_of_present}% số skill xuất hiện); "
                   f"chuẩn hóa cỡ {r.rarefy_n_used} JD: {r.n_rare_rarefied_mean} [{r.n_rare_rarefied_ci_low}; {r.n_rare_rarefied_ci_high}]; "
                   f"lượt nhắc hiếm {r.share_mentions_rare_pct}%; đuôi dài TB {r.mean_n_tail_skills}/JD")

    rep.append(f"\n=== Kỹ năng theo level: nhóm thấp ({'+'.join(LOW_LEVELS)}) vs nhóm cao ({'+'.join(HIGH_LEVELS)}) ===")
    for role, (obs, pm, p95, nl, nh) in ctx["perm_info"].items():
        verdict = "vượt ngưỡng ngẫu nhiên (p95)" if obs > p95 else "không vượt ngưỡng ngẫu nhiên"
        rep.append(f"  {role:<15} thấp={nl}, cao={nh}: skill only_high quan sát={obs}, kỳ vọng ngẫu nhiên={pm:.1f} (p95={p95:.0f}) -> {verdict}")
        sub = lvl[lvl.role == role]
        top = sub[sub.diff_pp > 0].sort_values("diff_pp", ascending=False).head(a.top_n)
        for s in top.itertuples():
            tags = ("only_high " if s.only_high else "") + ("hiếm " if s.is_rare_in_role else "") + ("khó " if s.is_hard_manual else "")
            rep.append(f"      {s.skill:<22} thấp {s.pct_low:>5.1f}% -> cao {s.pct_high:>5.1f}% (+{s.diff_pp:.1f} điểm %), p_holm={s.p_holm_in_role} {tags}")
    for sk in ctx["skipped"]:
        rep.append(f"  Bỏ qua (nhóm quá nhỏ): {sk}")

    rep += ["", "=== Hạn chế cần ghi trong khóa luận ===",
            "  - Nguồn tuyển dụng: dữ liệu gộp nhiều nền tảng, mỗi nền tảng ghi số skill và gắn tag khác nhau, cơ cấu nền tảng cũng khác giữa các nghề.",
            "    Độ rộng chịu một phần ảnh hưởng từ phong cách đăng tin của nền tảng; nghiên cứu không tách riêng yếu tố này.",
            "  - Độ rộng chỉ được mô tả và đo xu hướng theo level trong từng nghề, không kiểm định / xếp hạng giữa các nghề.",
            "  - level_group lấy từ nhãn level của JD; ô Lead+ ở một số nghề (vd. Frontend) có rất ít JD.",
            "  - n_skill_groups đếm số nhóm taxonomy (nhóm không cân bằng kích thước), chỉ là thước đo gần đúng của độ phức tạp tech stack.",
            "  - 'Hiếm' tính theo nghề (< ngưỡng % trong nghề) và phụ thuộc cỡ mẫu nghề; số skill hiếm được chuẩn hóa bằng lấy mẫu con cùng cỡ.",
            "  - Danh sách kỹ năng 'khó' là phân loại thủ công theo kiến thức chuyên ngành.",
            "  - Skill 'chỉ có ở level cao' dễ xuất hiện do ngẫu nhiên khi nhóm cao nhỏ: luôn đối chiếu với đường cơ sở hoán vị.",
            "  - JD có thể trùng theo công ty, CI có thể hẹp hơn thực tế."]
    return rep


# MAIN: chạy lần lượt Bước 1 -> 8
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="jd_skill_binary.csv")
    ap.add_argument("--outdir", default="rq3_out")
    ap.add_argument("--roles", default=",".join(TARGET_ROLES))
    ap.add_argument("--domain-cols", default=DEFAULT_DOMAIN_COLS, help="nhãn lĩnh vực loại khỏi danh sách công nghệ (như RQ1)")
    ap.add_argument("--taxonomy", default="skill_taxonomy.csv", help="dùng để điều chỉnh n_skill_groups khi bỏ nhãn lĩnh vực (tùy chọn)")
    ap.add_argument("--implied-pairs", default="", help="đường dẫn skill_implied_pairs.csv (mặc định: cạnh file input)")
    ap.add_argument("--hard-skills", default="", help="CSV danh sách kỹ năng khó (cột 'skill'); mặc định dùng đề xuất khởi đầu trong code")
    ap.add_argument("--rare-pct", type=float, default=5.0, help="skill hiếm: tần suất < ngưỡng %% trong JD của nghề")
    ap.add_argument("--min-cell", type=int, default=15, help="ô nghề x level có ít JD hơn mức này bị gắn cờ cell_n_small")
    ap.add_argument("--rarefy-n", type=int, default=60, help="cỡ mẫu con để chuẩn hóa số skill hiếm giữa các nghề")
    ap.add_argument("--n-rarefy", type=int, default=200, help="số lần lấy mẫu con")
    ap.add_argument("--n-boot", type=int, default=500, help="số lần bootstrap cho CI của Spearman")
    ap.add_argument("--n-perm", type=int, default=1000, help="số lần hoán vị nhãn level cho đường cơ sở only_high")
    ap.add_argument("--min-total", type=int, default=5, help="skill cần xuất hiện ở >= bấy nhiêu JD của nghề mới đưa vào rq3_level_skills.csv")
    ap.add_argument("--min-high", type=int, default=3, help="skill only_high cần xuất hiện ở >= bấy nhiêu JD nhóm cao")
    ap.add_argument("--min-group-n", type=int, default=10, help="nhóm thấp/cao của nghề cần >= bấy nhiêu JD mới phân tích")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-n", type=int, default=8, help="số skill hiển thị mỗi nghề trong báo cáo")
    ap.add_argument("--report", action="store_true", help="ghi thêm rq3_report.txt")
    return ap.parse_args()


def main():
    a = parse_args()
    roles = [r.strip() for r in a.roles.split(",") if r.strip()]
    domain_cols = parse_domain_cols(a.domain_cols)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    # ---- BƯỚC 1: nạp và chuẩn bị
    group_map = load_group_map(a.taxonomy)
    df, skill_cols = load_data(a.input, domain_cols, group_map)
    missing = [r for r in roles if r not in df.expertise_category.unique()]
    if missing:
        raise SystemExit(f"Không tìm thấy vị trí trong expertise_category: {missing}")
    implied, _ = resolve_implied(a.implied_pairs, a.input, df, skill_cols)
    df = add_dedup(df, implied)
    hard_cols, hard_missing, hard_src = resolve_hard_skills(a.hard_skills, skill_cols)
    print(f"[Bước 1] {len(skill_cols)} cột công nghệ, {len(implied)} cặp cha-con, {len(hard_cols)} kỹ năng khó thủ công ({hard_src})")

    # ---- BƯỚC 2: lọc JD và trục level
    g, n_noise = filter_jobs(df, roles)
    n_no_level = int((~g.level_group.isin(LEVELS)).sum())
    print(f"[Bước 2] {len(g)} JD has_hard (loại {n_noise} JD noise); {n_no_level} JD thiếu level_group sẽ bị bỏ ở phân tích theo level")

    # ---- BƯỚC 3: hiếm theo nghề + đặc trưng từng JD
    rare_by_role = define_rare(g, skill_cols, roles, a.rare_pct)
    g = add_jd_features(g, rare_by_role, hard_cols)
    g_lv = g[g.level_group.isin(LEVELS)].copy()
    exp_by_level = g_lv.groupby("level_group").exp_min.agg(["count", "mean", "median"]).reindex(LEVELS)
    print("[Bước 3] skill hiếm theo nghề: " + ", ".join(f"{r} {len(rare_by_role[r])}" for r in roles))

    # ---- BƯỚC 4: xu hướng theo level
    trend = trend_table(g_lv, roles, a.n_boot, a.seed)
    print("[Bước 4] xu hướng theo level xong")

    # ---- BƯỚC 5: bảng nghề x level
    by_level = by_level_table(g_lv, roles, trend, a.min_cell)
    print(f"[Bước 5] {len(by_level)} dòng nghề x level x chỉ số")

    # ---- BƯỚC 6: kỹ năng hiếm của nghề
    rare = rare_summary(g, skill_cols, roles, rare_by_role, hard_cols, a.rarefy_n, a.n_rarefy, a.rare_pct, a.seed)
    print("[Bước 6] số skill hiếm mỗi nghề (chuẩn hóa cỡ mẫu): " + ", ".join(f"{r.role} {r.n_rare_rarefied_mean}" for r in rare.itertuples()))

    # ---- BƯỚC 7: kỹ năng theo level
    lvl, skipped, perm_info = level_skills_table(g_lv, skill_cols, roles, rare_by_role, hard_cols,
                                                 a.min_total, a.min_high, a.min_group_n, a.n_perm, a.seed)
    print("[Bước 7] only_high quan sát / kỳ vọng ngẫu nhiên: "
          + ", ".join(f"{r} {o}/{m:.1f}" for r, (o, m, _, _, _) in perm_info.items())
          + (f"; bỏ qua {', '.join(skipped)}" if skipped else ""))

    # ---- BƯỚC 8: xuất file
    by_level.to_csv(out / "rq3_by_level.csv", index=False, encoding="utf-8-sig")
    rare.to_csv(out / "rq3_rare_skills.csv", index=False, encoding="utf-8-sig")
    lvl.to_csv(out / "rq3_level_skills.csv", index=False, encoding="utf-8-sig")
    files = ["rq3_by_level.csv", "rq3_rare_skills.csv", "rq3_level_skills.csv"]
    if a.report:
        ctx = {"a": a, "roles": roles, "by_level": by_level, "trend": trend, "rare": rare, "lvl": lvl, "perm_info": perm_info,
               "skipped": skipped, "domain_cols": domain_cols, "n_noise": n_noise, "n_no_level": n_no_level, "n_jobs": len(g_lv),
               "n_groups_adjusted": int(g_lv.groups_adjusted.sum()), "has_group_map": bool(group_map),
               "hard_cols": hard_cols, "hard_missing": hard_missing, "hard_src": hard_src, "exp_by_level": exp_by_level}
        rep = build_report(ctx)
        (out / "rq3_report.txt").write_text("\n".join(rep), encoding="utf-8")
        files.append("rq3_report.txt")
        print("\n".join(rep))
    print(f"[Bước 8] đã ghi vào {out}/: " + ", ".join(files))


if __name__ == "__main__":
    main()
