"""
rq1_analysis.py  -  Câu hỏi nghiên cứu 1

"Tổ hợp kỹ năng chuyên môn cốt lõi (Core Stack) của từng vị trí (Backend, Frontend, Data Engineer,
 Data Scientist, AI Engineer) gồm những công nghệ nào?"

THUẬT TOÁN
Bước 1: Bỏ ngoại ngữ và 5 nhãn cấp lĩnh vực AI/ML (AI, ML, DL, NLP, CV) khỏi danh sách công nghệ.
Bước 2: Tỷ lệ của skill trong vị trí = % số JD có skill / số JD của vị trí.
Bước 3: Phân hai tầng: core khi ≥ 30%, common khi ≥ 15%. Khoảng tin cậy Wilson 95% chỉ báo kèm, không dùng để loại skill.
Bước 4: Tổ hợp trong tập core + common:
- Cặp: Tính P(B|A), P(A|B), Jaccard, phi, tỷ lệ kỳ vọng nếu độc lập, chênh lệch thực tế trừ kỳ vọng. Loại bỏ 1 số cặp con-cha theo luật roll-up (skill_implied_pairs.csv). Gắn nhãn ứng viên: cặp xuất hiện nhiều, cặp có thể thay thế nhau (phi âm).
- Bộ ba: support và kỳ vọng nếu chỉ có tương tác cặp (IPF).
- Tiến hành dán nhãn các cặp: cặp xuất hiện nhiều, cặp có thể thay thế nhau (phi âm), bộ ba xuất hiện nhiều, cặp tỷ lệ thấp hầu như không có ý nghĩa để trống

Chạy: python rq1_analysis.py --input jd_skill_binary.csv --outdir rq1_out [--report --extras]
"""
import argparse
import re
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

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

CONCEPT_COLS = {"skill_mlops", "skill_data_lake", "skill_data_warehouse", "skill_microservices", "skill_event_driven"}

COMBO_COLS = ["role", "size", "skills", "n", "pct", "ci_low_pct", "ci_high_pct",
              "expected_pct", "excess_pp", "expected_pairwise_pct", "excess_pairwise_pp",
              "p_b_given_a", "p_a_given_b", "jaccard", "phi", "expected_n_both",
              "loai", "nhom_thu_cong", "ghi_chu"]


# BƯỚC 3  CORE STACK HAI TẦNG THEO VỊ TRÍ
def core_stack_table(g, skill_cols, roles, core_thr, common_thr, small_n, children_of):
    rows = []
    for role in roles:
        gr = g[g.expertise_category == role]
        n = len(gr)
        if n == 0:
            continue
        counts = gr[skill_cols].sum()
        for col in skill_cols:
            k = int(counts[col])
            pct = 100 * k / n
            lo, hi = wilson_ci(k, n)
            tier = "core" if pct >= core_thr else ("common" if pct >= common_thr else "no")
            rows.append({"role": role, "n_jobs": n, "skill": col, "n": k, "pct": round(pct, 1),
                         "ci_low_pct": round(100 * lo, 1), "ci_high_pct": round(100 * hi, 1),
                         "tier": tier, "is_concept": col in CONCEPT_COLS,
                         "implied_children": ", ".join(children_of.get(col, [])),
                         "role_n_below_small_n": n < small_n})
    out = pd.DataFrame(rows)
    order = {r: i for i, r in enumerate(roles)}
    out["_o"] = out.role.map(order)
    return out.sort_values(["_o", "pct"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


# BƯỚC 4  TỔ HỢP TRONG TẬP CORE + COMMON
def select_combo_skills(tax, role, max_skills):
    """4a: skill tầng core/common của vị trí, tối đa max_skills skill phổ biến nhất."""
    sub = tax[(tax.role == role) & (tax.tier.isin(["core", "common"]))].sort_values("pct", ascending=False)
    return sub.skill.tolist()[:max_skills]


def pair_table(X, names, role, min_pair_n, imp):
    """4b: mọi cặp trong tập skill. X: ma trận 0/1 (JD x skill). Trả về (bảng cặp, ma trận P(cột|hàng) %)."""
    n = X.shape[0]
    M = X.T @ X                                   # M[i, j] = số JD có cả skill i và j; đường chéo = n của từng skill
    k = np.diag(M).astype(float)
    rows = []
    for i, j in combinations(range(len(names)), 2):
        nb = int(M[i, j])
        pa, pb = k[i] / n, k[j] / n
        denom = np.sqrt(k[i] * (n - k[i]) * k[j] * (n - k[j]))
        lo, hi = wilson_ci(nb, n)
        rows.append({
            "role": role, "size": 2, "skills": f"{names[i]} + {names[j]}",
            "n": nb, "pct": round(100 * nb / n, 1),
            "ci_low_pct": round(100 * lo, 1), "ci_high_pct": round(100 * hi, 1),
            "expected_pct": round(100 * pa * pb, 1),                         # kỳ vọng nếu hai skill độc lập
            "excess_pp": round(100 * nb / n - 100 * pa * pb, 1),             # thực tế - kỳ vọng
            "p_b_given_a": round(100 * nb / k[i], 1), "p_a_given_b": round(100 * nb / k[j], 1),
            "jaccard": round(nb / (k[i] + k[j] - nb), 3),
            "phi": round((n * nb - k[i] * k[j]) / denom, 3) if denom > 0 else np.nan,
            "expected_n_both": round(k[i] * k[j] / n, 1),
            "exp_n_enough": k[i] * k[j] / n >= min_pair_n,
            "implied_by_rule": (i, j) in imp,
        })
    cond = M / k[:, None] * 100                   # cond[i, j] = P(skill_j | skill_i)
    return pd.DataFrame(rows), cond


def pairwise_expected_triple(Xs, iters=200):
    """Kỳ vọng ô (1,1,1) của bảng 2x2x2 dưới mô hình CHỈ có tương tác từng cặp (không có tương tác bậc 3),"""
    n = Xs.shape[0]
    T = np.zeros((2, 2, 2))
    for a, b, c in Xs:
        T[a, b, c] += 1
    marg = {(0, 1): T.sum(axis=2), (0, 2): T.sum(axis=1), (1, 2): T.sum(axis=0)}
    E = np.full((2, 2, 2), n / 8.0)
    for _ in range(iters):
        for (i, j), target in marg.items():
            axis = ({0, 1, 2} - {i, j}).pop()
            cur = E.sum(axis=axis)
            ratio = np.divide(target, cur, out=np.zeros_like(target), where=cur > 0)
            E = E * np.expand_dims(ratio, axis)
    return float(E[1, 1, 1] / n)


def triple_table(X, names, role, support_pct, imp):
    """4c: bộ ba có support >= support_pct, bỏ bộ chứa cặp con-cha theo luật."""
    n = X.shape[0]
    p = X.mean(axis=0)
    rows = []
    for idx in combinations(range(len(names)), 3):
        if any(pr in imp for pr in combinations(idx, 2)):
            continue
        cnt = int(X[:, list(idx)].all(axis=1).sum())
        pct = 100 * cnt / n
        if pct < support_pct:
            continue
        lo, hi = wilson_ci(cnt, n)
        exp = 100 * float(np.prod(p[list(idx)]))
        exp_pair = 100 * pairwise_expected_triple(X[:, list(idx)].astype(int))
        rows.append({"role": role, "size": 3, "skills": " + ".join(names[i] for i in idx),
                     "n": cnt, "pct": round(pct, 1),
                     "ci_low_pct": round(100 * lo, 1), "ci_high_pct": round(100 * hi, 1),
                     "expected_pct": round(exp, 1), "excess_pp": round(pct - exp, 1),
                     "expected_pairwise_pct": round(exp_pair, 1), "excess_pairwise_pp": round(pct - exp_pair, 1)})
    return pd.DataFrame(rows)


def assemble_combos(pairs, triples, support_pct, min_abs_phi, top_k):
    """4d + 4e: gắn nhãn ứng viên rồi gộp cặp và bộ ba thành một bảng."""
    p = pairs[~pairs.implied_by_rule].copy()
    p["loai"] = ""

    def tag(df, idx, label):
        df.loc[idx, "loai"] = (df.loc[idx, "loai"] + ";" + label).str.strip(";")

    tag(p, p.sort_values("pct", ascending=False).head(top_k).index, "cap_xuat_hien_nhieu")
    subs = p[(p.phi <= -min_abs_phi) & (p.exp_n_enough == True)]            # noqa: E712
    tag(p, subs.sort_values("phi").head(top_k).index, "cap_ung_vien_thay_the")
    p = p[(p.pct >= support_pct) | (p.loai != "")]                          # 4e: support đủ lớn HOẶC là ứng viên

    t = triples.copy()
    if not t.empty:
        t["loai"] = ""
        tag(t, t.sort_values("pct", ascending=False).head(top_k).index, "bo_ba_xuat_hien_nhieu")

    out = pd.concat([p, t], ignore_index=True)
    out["nhom_thu_cong"] = ""
    out["ghi_chu"] = ""
    return out.reindex(columns=COMBO_COLS).sort_values(["size", "pct"], ascending=[True, False]).reset_index(drop=True)





# BƯỚC 5  KIỂM TRA CHẤT LƯỢNG, BÁO CÁO VÀ XUẤT FILE
def find_hidden_pairs(df, skill_cols, implied, min_n, min_pct):
    """5a: cặp (A -> B) có P(B|A) >= min_pct trên các job has_hard (mọi nhóm nghề), ít nhất min_n job có A,"""
    h = df[df.has_skills == 1]
    if "skills_status" in h.columns:
        h = h[h.skills_status == "has_hard"]
    X = h[skill_cols].to_numpy(dtype=int)
    M = X.T @ X
    k = np.diag(M)
    known = set(implied)
    rows = []
    for i, a in enumerate(skill_cols):
        if k[i] < min_n:
            continue
        for j, b in enumerate(skill_cols):
            if i != j and 100 * M[i, j] / k[i] >= min_pct and (a, b) not in known:
                rows.append({"skill_a": short(a), "implies_b": short(b), "n_a": int(k[i]), "n_both": int(M[i, j]),
                             "pct_b_given_a": round(100 * M[i, j] / k[i], 1), "n_b": int(k[j])})
    return pd.DataFrame(rows, columns=["skill_a", "implies_b", "n_a", "n_both", "pct_b_given_a", "n_b"]).sort_values(
        ["pct_b_given_a", "n_a"], ascending=False).reset_index(drop=True)


def build_report(ctx):
    a, roles, tax, combos, hidden = ctx["a"], ctx["roles"], ctx["tax"], ctx["combos"], ctx["hidden"]
    rep = [f"Vị trí: {', '.join(roles)}",
           f"Nhãn lĩnh vực đã loại khỏi danh sách công nghệ: {', '.join(short(c) for c in ctx['domain_cols']) or '(không)'}",
           f"Tầng core: >= {a.core_threshold:.0f}% JD của vị trí; tầng common: >= {a.common_threshold:.0f}% (giá trị điểm, CI chỉ để tham khảo)",
           f"Lọc: has_skills = 1 và skills_status = has_hard; đã loại {ctx['n_noise']} JD chỉ có noise",
           f"Luật cha-con: {ctx['n_implied']} cặp con -> cha (nguồn: {ctx['implied_src']}); tỷ lệ skill cha gồm cả JD chỉ ghi skill con",
           ""]
    for role in roles:
        t = tax[tax.role == role]
        if t.empty:
            continue
        n = int(t.n_jobs.iloc[0])
        flag = "   [CẢNH BÁO: cỡ mẫu nhỏ]" if n < a.small_n else ""
        rep.append(f"{'=' * 8} {role}: n = {n} JD{flag}")
        for tier, cap in (("core", 20), ("common", a.top_n)):
            tt = t[t.tier == tier]
            rep.append(f"  Tầng {tier}: {len(tt)} skill" + (f" (hiển thị {cap} đầu)" if len(tt) > cap else ""))
            for s in tt.head(cap).itertuples():
                note = ("  [khái niệm]" if s.is_concept else "") + (f"  [gồm cả JD chỉ ghi: {s.implied_children}]" if s.implied_children else "")
                rep.append(f"    {short(s.skill):<26} {s.pct:>5.1f}%  (CI95 {s.ci_low_pct:.1f}-{s.ci_high_pct:.1f}%, n={s.n}){note}")
        c = combos[combos.role == role]
        if c.empty:
            rep.append("  Tổ hợp: không đủ skill ở tầng core/common để tính")
        else:
            rep.append(f"  Tổ hợp (support >= {a.combo_support:.0f}%), top {a.top_n} mỗi cỡ; excess = support thực tế - kỳ vọng nếu độc lập:")
            for size in (3, 2):
                for s in c[c["size"] == size].sort_values("pct", ascending=False).head(a.top_n).itertuples():
                    extra = f", kỳ vọng từ các cặp {s.expected_pairwise_pct:.1f}%, excess so với cặp {s.excess_pairwise_pp:+.1f}" if size == 3 else ""
                    rep.append(f"    [{size} skill] {s.skills}  {s.pct:.1f}% (n={s.n}, kỳ vọng {s.expected_pct:.1f}%, excess {s.excess_pp:+.1f}{extra})")
            for s in c[c.loai.str.contains("thay_the")].sort_values("phi").head(3).itertuples():
                rep.append(f"    [ứng viên thay thế] {s.skills}: phi={s.phi}, n_both={s.n}, kỳ vọng {s.expected_n_both} JD")
        rep.append("")
    rep.append(f"=== Cặp có P(B|A) >= {a.hidden_pair_pct:.0f}% (n_A >= {a.hidden_pair_min_n}) nhưng KHÔNG thuộc luật roll-up: {len(hidden)} cặp ===")
    for s in hidden.head(a.top_n).itertuples():
        rep.append(f"  {s.skill_a} -> {s.implies_b}: {s.pct_b_given_a:.1f}% (n_A={s.n_a}, n_both={s.n_both}); cần kiểm tra: phụ thuộc thật hay luật ngầm chưa khai báo")
    rep += ["", "=== Hạn chế cần ghi trong khóa luận ===",
            "  - Nguồn tuyển dụng: dữ liệu gộp nhiều nền tảng, mỗi nền tảng gắn tag skill khác nhau và cơ cấu nền tảng cũng khác giữa các vị trí.",
            "    Core stack vì vậy chịu một phần ảnh hưởng từ phong cách đăng tin của nền tảng; nghiên cứu không tách riêng yếu tố này.",
            "  - Cỡ mẫu một số vị trí nhỏ (vd. Data Scientist): chỉ mô tả.",
            "  - 'Cốt lõi' theo tần suất trong JD, không phân biệt bắt buộc và ưu tiên.",
            "  - Phân nhóm tổ hợp (Core Standard / Advanced / Substitutes) là phân loại thủ công theo kiến thức chuyên ngành.",
            "  - Skill cha (Spring, CI/CD, GCP...) được tính cả khi JD chỉ ghi skill con theo luật roll-up; các cặp con-cha không được dùng làm bằng chứng đồng xuất hiện.",
            "  - JD có thể trùng theo công ty, CI có thể hẹp hơn thực tế."]
    return rep


# MAIN: chạy lần lượt Bước 1 -> 5
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="jd_skill_binary.csv")
    ap.add_argument("--outdir", default="rq1_out")
    ap.add_argument("--roles", default=",".join(TARGET_ROLES))
    ap.add_argument("--domain-cols", default=DEFAULT_DOMAIN_COLS,
                    help="nhãn lĩnh vực loại khỏi danh sách công nghệ, cách nhau bằng dấu phẩy (có hoặc không tiền tố skill_); để trống nếu không loại")
    ap.add_argument("--implied-pairs", default="", help="đường dẫn skill_implied_pairs.csv (mặc định: cạnh file input)")
    ap.add_argument("--core-threshold", type=float, default=30.0, help="%% JD tối thiểu của tầng core")
    ap.add_argument("--common-threshold", type=float, default=15.0, help="%% JD tối thiểu của tầng common")
    ap.add_argument("--small-n", type=int, default=100, help="vị trí có ít JD hơn mức này bị gắn cờ cỡ mẫu nhỏ")
    ap.add_argument("--max-combo-skills", type=int, default=20, help="số skill tối đa (phổ biến nhất) đưa vào tính tổ hợp mỗi vị trí")
    ap.add_argument("--combo-support", type=float, default=10.0, help="%% JD tối thiểu để giữ một cặp/bộ ba trong rq1_combos.csv")
    ap.add_argument("--min-pair-n", type=int, default=5, help="số JD kỳ vọng chứa cả hai tối thiểu để một cặp phi âm là ứng viên thay thế")
    ap.add_argument("--min-abs-phi", type=float, default=0.10, help="|phi| tối thiểu để một cặp phi âm thành ứng viên thay thế")
    ap.add_argument("--candidates-k", type=int, default=15, help="số ứng viên mỗi loại gắn nhãn trong cột loai")
    ap.add_argument("--hidden-pair-pct", type=float, default=99.0, help="P(B|A) %% từ mức này trở lên mà không thuộc luật thì liệt kê để kiểm tra")
    ap.add_argument("--hidden-pair-min-n", type=int, default=20, help="số job tối thiểu có skill A khi kiểm tra cặp ẩn")
    ap.add_argument("--top-n", type=int, default=10, help="số mục hiển thị trong báo cáo")
    ap.add_argument("--keep-all", action="store_true", help="ghi cả skill tầng 'no' vào rq1_core_stack.csv")
    ap.add_argument("--report", action="store_true", help="ghi thêm rq1_report.txt")
    ap.add_argument("--extras", action="store_true", help="ghi thêm heatmap và rq1_hidden_implied_pairs.csv")
    return ap.parse_args()


def main():
    a = parse_args()
    roles = [r.strip() for r in a.roles.split(",") if r.strip()]
    domain_cols = parse_domain_cols(a.domain_cols)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    # ---- BƯỚC 1: nạp và chuẩn bị
    df, skill_cols = load_data(a.input, domain_cols)
    missing = [r for r in roles if r not in df.expertise_category.unique()]
    if missing:
        raise SystemExit(f"Không tìm thấy vị trí trong expertise_category: {missing}")
    implied, implied_src = resolve_implied(a.implied_pairs, a.input, df, skill_cols)
    imp_names = {(short(c), short(p)) for c, p in implied}
    children_of = {}
    for c, p in implied:
        children_of.setdefault(p, []).append(short(c))
    print(f"[Bước 1] {len(skill_cols)} cột công nghệ (đã bỏ {len(domain_cols)} nhãn lĩnh vực), {len(implied)} cặp cha-con")

    # ---- BƯỚC 2: lọc job
    g, n_noise = filter_jobs(df, roles)
    print(f"[Bước 2] {len(g)} JD của {len(roles)} vị trí (loại {n_noise} JD chỉ có noise)")

    # ---- BƯỚC 3: core stack
    tax = core_stack_table(g, skill_cols, roles, a.core_threshold, a.common_threshold, a.small_n, children_of)
    print(f"[Bước 3] core stack: {int((tax.tier == 'core').sum())} dòng core, {int((tax.tier == 'common').sum())} dòng common")

    # ---- BƯỚC 4: tổ hợp
    all_combos, heat_saved, skipped = [], [], []
    for role in roles:
        sk = select_combo_skills(tax, role, a.max_combo_skills)                                  # 4a
        if len(sk) < 2:
            skipped.append(role)
            continue
        names = [short(s) for s in sk]
        X = g.loc[g.expertise_category == role, sk].to_numpy(dtype=int)
        imp = {(i, j) for i, j in combinations(range(len(names)), 2)
               if (names[i], names[j]) in imp_names or (names[j], names[i]) in imp_names}
        pairs, cond = pair_table(X, names, role, a.min_pair_n, imp)                              # 4b
        triples = triple_table(X, names, role, a.combo_support, imp)                             # 4c
        all_combos.append(assemble_combos(pairs, triples, a.combo_support, a.min_abs_phi, a.candidates_k))   # 4d + 4e
    combos = pd.concat(all_combos, ignore_index=True) if all_combos else pd.DataFrame(columns=COMBO_COLS)
    print(f"[Bước 4] {len(combos)} dòng tổ hợp" + (f"; bỏ qua (thiếu skill): {', '.join(skipped)}" if skipped else ""))

    # ---- BƯỚC 5: kiểm tra chất lượng và xuất file
    hidden = find_hidden_pairs(df, skill_cols, implied, a.hidden_pair_min_n, a.hidden_pair_pct)
    (tax if a.keep_all else tax[tax.tier != "no"]).to_csv(out / "rq1_core_stack.csv", index=False, encoding="utf-8-sig")
    combos.to_csv(out / "rq1_combos.csv", index=False, encoding="utf-8-sig")
    files = ["rq1_core_stack.csv", "rq1_combos.csv"]
    if a.extras:
        hidden.to_csv(out / "rq1_hidden_implied_pairs.csv", index=False, encoding="utf-8-sig")
        files.append("rq1_hidden_implied_pairs.csv")
    if a.report:
        ctx = {"a": a, "roles": roles, "tax": tax, "combos": combos, "hidden": hidden, "domain_cols": domain_cols,
               "n_noise": n_noise, "n_implied": len(implied), "implied_src": implied_src}
        rep = build_report(ctx)
        (out / "rq1_report.txt").write_text("\n".join(rep), encoding="utf-8")
        files.append("rq1_report.txt")
        print("\n".join(rep))
    if len(hidden):
        print(f"[Cảnh báo] {len(hidden)} cặp P(B|A) >= {a.hidden_pair_pct:.0f}% chưa thuộc luật roll-up "
              f"(xem bằng --extras hoặc --report): " + ", ".join(f"{h.skill_a}->{h.implies_b}" for h in hidden.head(5).itertuples()))
    print(f"[Bước 5] đã ghi vào {out}/: " + ", ".join(files))


if __name__ == "__main__":
    main()
