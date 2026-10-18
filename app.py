"""Dashboard Streamlit cho 4 câu hỏi nghiên cứu (rq1_analysis, rq2_bara, rq3_analysis, rq4_geography).

Chỉ ĐỌC kết quả các script đã chạy (thư mục results/rq1_out ... rq4_out), nên số liệu trên dashboard
trùng khớp báo cáo. Riêng tab Tổng quan đọc thẳng jd_skill_binary.csv (nếu có).
Chỉ cần theme.py cạnh file này (không cần analytics.py, jd_loader.py).

Chạy:
    python -m streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import theme as T


st.set_page_config(page_title="Dashboard kỹ năng IT", layout="wide")
T.apply(st)
st.title("Dashboard kỹ năng IT theo nhóm nghề")   # hiện ngay, trước khi đọc dữ liệu

BASE = Path(__file__).parent
LEVELS = ["Entry", "Middle", "Senior", "Lead+"]
ALL_LV = "TẤT CẢ"
SENIOR_PLUS = "Senior+ (gộp 2 bậc trên)"
ALL_ROLES = "TẤT CẢ nhóm nghề"
BIN_METRICS = {"has_rare_skill", "has_hard_manual"}
METRIC_LABELS = {
    "n_hard_skills_dedup": "Độ rộng: số hard skill (đã loại trùng cha-con)",
    "n_hard_skills": "Độ rộng: số hard skill (chưa loại trùng)",
    "n_skill_groups": "Độ phức tạp: số nhóm kỹ năng",
    "n_rare_skills": "Số skill hiếm mỗi JD",
    "has_rare_skill": "% JD có ít nhất 1 skill hiếm",
    "n_hard_manual": "Số skill khó (danh sách thủ công) mỗi JD",
    "has_hard_manual": "% JD có ít nhất 1 skill khó",
    "n_tail_skills": "Số skill đuôi dài mỗi JD",
}


# ----------------------------------------------------------------------------- tìm và đọc file
def first_existing(*cands):
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def _default_results():
    p = first_existing(BASE / "results", Path.cwd() / "results", BASE.parent / "results")
    return str(p or BASE / "results")


# Đường dẫn tự dò (không còn ô nhập ở thanh bên). Muốn cố định thì sửa trực tiếp 3 dòng dưới.
RES = Path(_default_results())
JD_PATH = str(first_existing(BASE / "jd_skill_binary.csv", BASE / "data" / "jd_skill_binary.csv", RES.parent / "jd_skill_binary.csv") or "")
TAX_PATH = str(first_existing(BASE / "skill_taxonomy.csv", BASE / "data" / "skill_taxonomy.csv", RES.parent / "skill_taxonomy.csv") or "")


def rq_path(sub, fname):
    return first_existing(RES / sub / fname, RES / fname)


@st.cache_data(show_spinner=False)
def _read_csv(path, mtime):
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def _read_xlsx(path, mtime):
    return pd.read_excel(path, sheet_name=None)


def load_csv(sub, fname):
    p = rq_path(sub, fname)
    return None if p is None else _read_csv(str(p), p.stat().st_mtime)


def load_labels():
    if not TAX_PATH or not Path(TAX_PATH).is_file():
        return {}
    t = _read_csv(TAX_PATH, Path(TAX_PATH).stat().st_mtime)
    if "canonical" not in t.columns:
        return {}
    import re
    def slug(c):
        s = str(c).lower().replace("c++", "cpp").replace("c#", "csharp").replace(".net", "dotnet")
        return "skill_" + re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return {slug(c): str(c) for c in t.canonical}


LABELS = load_labels()


def lab(x):
    """'skill_python' hoặc 'python' -> tên đẹp (từ taxonomy nếu có)."""
    key = x if str(x).startswith("skill_") else f"skill_{x}"
    return LABELS.get(key) or str(x).replace("skill_", "").replace("_", " ")


def lab_combo(s):
    return " + ".join(lab(p.strip()) for p in str(s).split(" + "))


def as_bool(s):
    return s.astype(str).str.strip().str.lower().eq("true")


def with_err(d, pct="pct", lo="ci_low_pct", hi="ci_high_pct"):
    return d.assign(err_plus=(d[hi] - d[pct]).clip(lower=0), err_minus=(d[pct] - d[lo]).clip(lower=0))


def bar_h(n, per=24, base=140, lo=320):
    return max(lo, base + per * n)


def missing(script):
    st.info(f"Chưa có kết quả của {script} trong thư mục {RES}. Chạy `python run_all.py --input jd_skill_binary.csv "
            f"--taxonomy skill_taxonomy.csv` rồi tải lại trang.")


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)
    st.write("")


# ----------------------------------------------------------------------------- Tổng quan
def tab_overview():
    if not JD_PATH or not Path(JD_PATH).is_file():
        st.info("Chưa tìm thấy jd_skill_binary.csv (đặt cạnh app.py hoặc trong thư mục data/) nên chưa có số liệu tổng quan. Các trang RQ vẫn dùng được.")
        return
    jd = _read_csv(JD_PATH, Path(JD_PATH).stat().st_mtime)
    c = st.columns(4)
    c[0].metric("Số job", f"{len(jd):,}")
    if "expertise_category" in jd.columns:
        c[1].metric("Số nhóm nghề", int(jd.expertise_category.nunique()))
    if "skills_status" in jd.columns:
        c[2].metric("Job có hard skill", f"{100 * (jd.skills_status == 'has_hard').mean():.1f}%")
    if "city" in jd.columns:
        c[3].metric("Số thành phố", int(jd.city.nunique()))
    l, r = st.columns(2)
    if "expertise_category" in jd.columns:
        v = jd.expertise_category.value_counts().rename_axis("role").reset_index(name="n").iloc[::-1]
        l.plotly_chart(px.bar(v, x="n", y="role", orientation="h", title="Số job theo nhóm nghề",
                              labels={"n": "Số job", "role": "Nhóm nghề"}, height=bar_h(len(v))))
    if "source" in jd.columns:
        v = jd.source.value_counts().rename_axis("source").reset_index(name="n")
        r.plotly_chart(px.bar(v, x="source", y="n", title="Số job theo nguồn", labels={"n": "Số job", "source": "Nguồn"}))
    if {"expertise_category", "source"} <= set(jd.columns):
        x = jd.groupby(["expertise_category", "source"]).size().rename("n").reset_index()
        st.plotly_chart(px.bar(x, x="expertise_category", y="n", color="source", barmode="stack",
                               labels={"expertise_category": "Nhóm nghề", "n": "Số job", "source": "Nguồn"},
                               title="Cơ cấu nguồn của từng nhóm nghề"))
        note("Nhóm nghề và nguồn tuyển dụng gần như trùng nhau (mỗi nguồn nghiêng về vài nhóm nghề), nên khác biệt giữa nhóm nghề "
             "có thể lẫn với cách đăng tin của từng nền tảng. Đây là hạn chế của đề tài.")


# ----------------------------------------------------------------------------- RQ1: core stack
def tab_rq1():
    core = load_csv("rq1_out", "rq1_core_stack.csv")
    if core is None:
        return missing("rq1_analysis.py")
    combos = load_csv("rq1_out", "rq1_combos.csv")
    core = core.assign(skill_label=core.skill.map(lab))
    st.caption("Core stack = skill xuất hiện ở ít nhất 30% JD của vị trí (tầng core), 15% (tầng common). "
               "Thanh sai số là khoảng tin cậy Wilson 95%. Tỷ lệ của skill cha (Spring, CI/CD, AWS...) gồm cả JD chỉ ghi skill con.")
    roles = list(core.role.drop_duplicates())
    role = st.selectbox("Vị trí", roles, key="rq1_role")
    sub = core[core.role == role]
    n = int(sub.n_jobs.iloc[0])
    if "role_n_below_small_n" in sub.columns and bool(as_bool(sub.role_n_below_small_n).iloc[0]):
        st.warning(f"{role} chỉ có {n} JD có hard skill: tỷ lệ dao động mạnh, đọc kèm khoảng tin cậy và chỉ mô tả.")
    c1, c2 = st.columns(2)
    tiers = c1.multiselect("Tầng hiển thị", ["core", "common"], default=["core", "common"], key="rq1_tiers")
    top = c2.slider("Số skill", 5, 40, 20, key="rq1_top")
    view = sub[sub.tier.isin(tiers)].sort_values("pct", ascending=False).head(top)
    if view.empty:
        st.info("Không có skill nào ở tầng đã chọn.")
    else:
        st.plotly_chart(px.bar(with_err(view).iloc[::-1], x="pct", y="skill_label", color="tier", orientation="h",
                               error_x="err_plus", error_x_minus="err_minus",
                               color_discrete_map={"core": T.DARK, "common": T.GREEN},
                               labels={"pct": "% JD yêu cầu", "skill_label": "Skill", "tier": "Tầng"},
                               title=f"{role}: n = {n} JD có hard skill", height=bar_h(len(view))))

    st.subheader("So sánh core stack giữa các vị trí")
    keep = core[core.tier == "core"].groupby("skill_label").pct.max().sort_values(ascending=False).head(30).index
    m = core[core.skill_label.isin(keep)].pivot_table(index="skill_label", columns="role", values="pct")
    m = m.reindex(index=list(keep), columns=roles)
    st.plotly_chart(px.imshow(m.round(1), aspect="auto", text_auto=".0f", color_continuous_scale=T.SEQ,
                              labels=dict(color="% JD"), height=bar_h(len(m), per=22)))
    st.caption("Ô trống = skill đó dưới ngưỡng common (15%) ở vị trí tương ứng.")

    st.subheader("Tổ hợp kỹ năng đi cùng nhau")
    if combos is None or combos.empty:
        st.info("Chưa có rq1_combos.csv hoặc không đủ skill ở tầng core/common để tính tổ hợp.")
    else:
        cr = combos[combos.role == role]
        c1, c2 = st.columns(2)
        size = c1.radio("Cỡ tổ hợp", [2, 3], horizontal=True, key="rq1_size",
                        format_func=lambda s: "Cặp (2 skill)" if s == 2 else "Bộ ba (3 skill)")
        topc = c2.slider("Số tổ hợp", 5, 30, 12, key="rq1_topc")
        cs = cr[cr["size"] == size].sort_values("pct", ascending=False).head(topc).copy()
        if cs.empty:
            st.info("Vị trí này không có tổ hợp đạt ngưỡng support.")
        else:
            cs["combo"] = cs.skills.map(lab_combo)
            series = {"pct": "Thực tế", "expected_pct": "Kỳ vọng nếu độc lập"}
            if size == 3:
                series["expected_pairwise_pct"] = "Kỳ vọng nếu chỉ có tương tác cặp"
            long = cs.melt(id_vars=["combo"], value_vars=list(series), var_name="loai", value_name="value")
            long["Loại"] = long["loai"].map(series)
            order = cs.combo.tolist()[::-1]
            st.plotly_chart(px.bar(long, x="value", y="combo", color="Loại", barmode="group", orientation="h",
                                   category_orders={"combo": order},
                                   color_discrete_sequence=[T.DARK, T.SAGE, T.TEAL],
                                   labels={"value": "% JD", "combo": "Tổ hợp"},
                                   title=f"{role}: tổ hợp phổ biến nhất (thực tế so với kỳ vọng)", height=bar_h(len(cs) * 2, per=16)))
            st.caption("Thực tế vượt kỳ vọng (excess_pp > 0) nghĩa là các skill này đi cùng nhau nhiều hơn mức ngẫu nhiên.")
        if "loai" in cr.columns:
            sub_c = cr[cr.loai.fillna("").str.contains("thay_the")].sort_values("phi").copy()
            if len(sub_c):
                sub_c["combo"] = sub_c.skills.map(lab_combo)
                with st.expander("Ứng viên thay thế (hai skill hiếm khi xuất hiện cùng nhau)"):
                    st.dataframe(sub_c[["combo", "n", "expected_n_both", "phi"]], hide_index=True)
    hid = load_csv("rq1_out", "rq1_hidden_implied_pairs.csv")
    if hid is not None and len(hid):
        with st.expander(f"Cặp P(B|A) rất cao nhưng chưa thuộc luật roll-up ({len(hid)} cặp)"):
            st.dataframe(hid, hide_index=True)


# ----------------------------------------------------------------------------- RQ2: mật độ
def tab_rq2():
    p = rq_path("rq2_out", "rq2_density.xlsx")
    if p is None:
        return missing("rq2_bara.py")
    try:
        sheets = _read_xlsx(str(p), p.stat().st_mtime)
    except ImportError:
        st.error("Cần thư viện openpyxl để đọc rq2_density.xlsx: chạy `python -m pip install openpyxl`.")
        return
    parts = []
    for name, label in (("Cloud_DevOps", "Cloud/DevOps"), ("AI_LLM", "AI/LLM")):
        d = sheets.get(name)
        if d is not None and len(d):
            parts.append(d.assign(metric=label, group="Công nghệ"))
    d = sheets.get("Ngoai_ngu")
    if d is not None and len(d):
        parts.append(d.assign(metric=d["language"], group="Ngoại ngữ"))
    if not parts:
        return missing("rq2_bara.py")
    dens = pd.concat(parts, ignore_index=True)
    dens["small"] = as_bool(dens.small_sample)
    dens = with_err(dens)
    st.caption("Mật độ = % JD của nhóm nghề yêu cầu ít nhất 1 skill thuộc nhóm. Cloud/DevOps và AI/LLM lấy mẫu số là JD có hard skill; "
               "ngoại ngữ lấy mẫu số là mọi JD của nhóm nghề. Thanh sai số là Wilson 95%.")
    note("Nhóm AI/LLM tính cả các nhãn lĩnh vực như machine learning, deep learning (khác RQ1 và RQ3 vốn loại các nhãn này), "
         "vì chính chúng là yêu cầu AI cần đo.")
    hide_small = st.checkbox("Ẩn nhóm nghề có ít JD (cỡ mẫu nhỏ)", True, key="rq2_hide")
    view = dens[~dens.small] if hide_small else dens
    if hide_small and dens.small.any():
        small_roles = sorted(dens[dens.small].role.unique())
        st.caption("Đã ẩn do cỡ mẫu nhỏ: " + ", ".join(small_roles))
    if view.empty:
        st.info("Không còn nhóm nghề nào sau khi ẩn cỡ mẫu nhỏ.")
        return
    n_by_role = dens.drop_duplicates("role").set_index("role").n_jobs
    order = [r for r in n_by_role.sort_values(ascending=False).index if r != ALL_ROLES]
    if ALL_ROLES in n_by_role.index:
        order.append(ALL_ROLES)
    order = [r for r in order if r in set(view.role)]
    m = view.pivot_table(index="role", columns="metric", values="pct").reindex(order)
    st.plotly_chart(px.imshow(m.round(1), aspect="auto", text_auto=".0f", color_continuous_scale=T.SEQ,
                              labels=dict(color="% JD"), height=bar_h(len(m), per=30)))
    metric = st.selectbox("Chỉ số", sorted(view.metric.unique()), key="rq2_metric")
    v = view[view.metric == metric].sort_values("pct")
    st.plotly_chart(px.bar(v, x="pct", y="role", orientation="h", error_x="err_plus", error_x_minus="err_minus",
                           hover_data=["n_matched", "n_denominator"],
                           labels={"pct": "% JD", "role": "Nhóm nghề"}, title=metric, height=bar_h(len(v))))
    langs = view[(view.group == "Ngoại ngữ") & (~view.metric.astype(str).str.startswith("Bất kỳ"))]
    if len(langs):
        st.plotly_chart(px.bar(langs, x="role", y="pct", color="metric", barmode="group",
                               error_y="err_plus", error_y_minus="err_minus",
                               labels={"role": "Nhóm nghề", "pct": "% JD", "metric": "Ngoại ngữ"},
                               title="Yêu cầu ngoại ngữ theo nhóm nghề"))


# ----------------------------------------------------------------------------- RQ3: level
def tab_rq3():
    bl = load_csv("rq3_out", "rq3_by_level.csv")
    if bl is None:
        return missing("rq3_analysis.py")
    rare = load_csv("rq3_out", "rq3_rare_skills.csv")
    lvl = load_csv("rq3_out", "rq3_level_skills.csv")
    bl = bl.assign(small=as_bool(bl.cell_n_small))

    st.subheader("Độ rộng và độ phức tạp theo level")
    st.caption("Level lấy từ level_group (Entry / Middle / Senior / Lead+), không đi qua số năm kinh nghiệm. "
               "Spearman đo xu hướng tăng theo level, hiệu chỉnh Holm giữa các nghề.")
    avail = [m for m in METRIC_LABELS if m in set(bl.metric)]
    c1, c2, c3 = st.columns(3)
    metric = c1.selectbox("Chỉ số", avail, format_func=METRIC_LABELS.get, key="rq3_metric")
    is_bin = metric in BIN_METRICS
    stat_name = "Tỷ lệ JD" if is_bin else c2.radio("Thống kê", ["Trung bình", "Trung vị"], horizontal=True, key="rq3_stat")
    stat = "mean" if is_bin or stat_name == "Trung bình" else "median"
    hide_small = c3.checkbox("Ẩn ô có ít JD", True, key="rq3_hide")
    d = bl[bl.metric == metric].copy()
    d["value"] = d[stat] * (100 if is_bin else 1)
    if is_bin:
        d["err_plus"] = ((d.ci_high - d["mean"]) * 100).clip(lower=0)
        d["err_minus"] = ((d["mean"] - d.ci_low) * 100).clip(lower=0)
    lv = d[d.level.isin(LEVELS)]
    if hide_small:
        lv = lv[~lv.small]
    ylab = "% JD" if is_bin else stat_name
    if lv.empty:
        st.info("Không có ô nghề x level nào đủ JD để vẽ. Bỏ chọn 'Ẩn ô có ít JD' để xem tất cả.")
    else:
        err = dict(error_y="err_plus", error_y_minus="err_minus") if is_bin else {}
        st.plotly_chart(px.line(lv, x="level", y="value", color="role", markers=True, category_orders={"level": LEVELS},
                                labels={"level": "Level", "value": ylab, "role": "Vị trí"},
                                title=METRIC_LABELS[metric], **err))
    st.caption("Ô có quá ít JD bị ẩn khỏi biểu đồ (ngưỡng đặt khi chạy rq3_analysis.py, mặc định 15 JD).")
    d["cell"] = d.value.round(2).astype(str) + " [n=" + d.n_jobs.astype(str) + "]"
    piv = d.pivot(index="role", columns="level", values="cell")
    st.dataframe(piv.reindex(columns=[c for c in [ALL_LV] + LEVELS if c in piv.columns]).fillna("-"))

    tr = d[d.level == ALL_LV][["role", "n_jobs", "trend_rho", "trend_ci_low", "trend_ci_high", "trend_p_holm"]].dropna(subset=["trend_rho"])
    if len(tr):
        st.markdown("**Xu hướng theo level (Spearman, mỗi nghề)**")
        t2 = tr.assign(err_plus=(tr.trend_ci_high - tr.trend_rho).clip(lower=0), err_minus=(tr.trend_rho - tr.trend_ci_low).clip(lower=0))
        st.plotly_chart(px.bar(t2, x="role", y="trend_rho", error_y="err_plus", error_y_minus="err_minus",
                               labels={"role": "Vị trí", "trend_rho": "Spearman rho (CI 95% bootstrap)"}))
        st.dataframe(tr.rename(columns={"role": "Vị trí", "n_jobs": "Số JD", "trend_rho": "rho", "trend_ci_low": "CI thấp",
                                        "trend_ci_high": "CI cao", "trend_p_holm": "p (Holm)"}), hide_index=True)

    st.subheader("Kỹ năng hiếm của từng nghề")
    if rare is None or rare.empty:
        st.info("Chưa có rq3_rare_skills.csv.")
    else:
        r2 = rare.assign(err_plus=(rare.n_rare_rarefied_ci_high - rare.n_rare_rarefied_mean).clip(lower=0),
                         err_minus=(rare.n_rare_rarefied_mean - rare.n_rare_rarefied_ci_low).clip(lower=0))
        l, r = st.columns(2)
        l.plotly_chart(px.bar(r2, x="role", y="n_rare_rarefied_mean", error_y="err_plus", error_y_minus="err_minus",
                              labels={"role": "Vị trí", "n_rare_rarefied_mean": "Số skill hiếm (cùng cỡ mẫu)"},
                              title="Số skill hiếm sau chuẩn hóa cỡ mẫu"))
        r.plotly_chart(px.bar(rare, x="role", y="share_mentions_rare_pct",
                              labels={"role": "Vị trí", "share_mentions_rare_pct": "% lượt nhắc là skill hiếm"},
                              title="Tỷ trọng lượt nhắc skill hiếm"))
        st.caption("Skill hiếm = xuất hiện ở dưới 5% JD của chính nghề đó (mặc định). Rarefaction lấy mẫu con cùng cỡ để so công bằng giữa nghề nhiều và ít JD.")
        cols = [c for c in ["role", "n_jobs", "n_skills_present", "n_rare_skills", "pct_rare_of_present", "n_rare_rarefied_mean",
                            "share_mentions_rare_pct", "n_hard_manual_present", "rare_skills"] if c in rare.columns]
        with st.expander("Bảng và danh sách skill hiếm"):
            st.dataframe(rare[cols], hide_index=True)

    st.subheader("Level cao có thêm kỹ năng nào so với level thấp?")
    if lvl is None or lvl.empty:
        st.info("Chưa có rq3_level_skills.csv, hoặc không nghề nào đủ JD ở cả nhóm thấp và cao.")
        return
    roles = list(lvl.role.drop_duplicates())
    role = st.selectbox("Vị trí", roles, key="rq3_role")
    s = lvl[lvl.role == role].copy()
    s["skill_label"] = s.skill.map(lab)
    s["only_high"] = as_bool(s.only_high)
    c = st.columns(3)
    c[0].metric("Skill chỉ có ở level cao", int(s.only_high.sum()))
    c[1].metric("Kỳ vọng nếu level không liên quan", f"{s.role_only_high_perm_mean.iloc[0]:.1f}")
    c[2].metric("Ngưỡng ngẫu nhiên P95", f"{s.role_only_high_perm_p95.iloc[0]:.1f}")
    if s.only_high.sum() > s.role_only_high_perm_p95.iloc[0]:
        note("Số skill chỉ có ở level cao vượt ngưỡng P95 của đường cơ sở hoán vị: khó giải thích chỉ bằng ngẫu nhiên.")
    else:
        note("Số skill chỉ có ở level cao không vượt ngưỡng P95 của đường cơ sở hoán vị: chưa đủ bằng chứng là khác ngẫu nhiên.")
    top = st.slider("Số skill hiển thị", 5, 30, 15, key="rq3_topskill")
    only_sig = st.checkbox("Chỉ skill có p hiệu chỉnh Holm < 0.05", False, key="rq3_sig")
    v = s[s.p_holm_in_role < 0.05] if only_sig else s
    v = v.sort_values("diff_pp", ascending=False).head(top)
    if v.empty:
        st.info("Không có skill nào thỏa điều kiện.")
    else:
        st.plotly_chart(px.bar(v.iloc[::-1], x="diff_pp", y="skill_label", orientation="h", color="only_high",
                               color_discrete_map={True: T.DARK, False: T.SAGE},
                               hover_data=["pct_low", "pct_high", "p_holm_in_role"],
                               labels={"diff_pp": "Chênh lệch điểm % (cao trừ thấp)", "skill_label": "Skill", "only_high": "Chỉ có ở level cao"},
                               title=f"{role}: skill nghiêng về level cao (Senior + Lead+ so với Entry + Middle)", height=bar_h(len(v))))
        heat = v.set_index("skill_label")[[c for c in ["pct_Entry", "pct_Middle", "pct_Senior", "pct_Lead+"] if c in v.columns]]
        heat.columns = [c.replace("pct_", "") for c in heat.columns]
        st.plotly_chart(px.imshow(heat.round(1), aspect="auto", text_auto=".0f", color_continuous_scale=T.SEQ,
                                  labels=dict(color="% JD"), height=bar_h(len(heat), per=22)))
        with st.expander("Bảng chi tiết"):
            st.dataframe(v[["skill_label", "n_total", "pct_low", "pct_high", "diff_pp", "fisher_p", "p_holm_in_role",
                            "only_high", "is_rare_in_role", "is_hard_manual"]], hide_index=True)


# ----------------------------------------------------------------------------- RQ4: địa lý
def tab_rq4():
    d = load_csv("rq4_out", "rq4_city_level.csv")
    if d is None:
        return missing("rq4_geography.py")
    d = d.assign(reliable=as_bool(d.city_sample_reliable))
    g = d[d.level == SENIOR_PLUS]
    ok = g[g.reliable]
    st.caption("Senior+ gồm Senior, Lead/Principal và Manager+. Chỉ tính job có cả city và level; đã loại 'Không rõ' và 'Remote'. "
               "Hai cách đọc 'tập trung': (a) số job tuyệt đối, (b) tỷ lệ Senior+ trong job của chính thành phố đó.")
    if ok.empty:
        st.info("Không có thành phố nào đủ cỡ mẫu.")
        return
    topn = st.slider("Số thành phố", 3, 30, 10, key="rq4_top")
    l, r = st.columns(2)
    a = ok.sort_values("n_jobs_in_level", ascending=False).head(topn).sort_values("n_jobs_in_level")
    l.plotly_chart(px.bar(a, x="n_jobs_in_level", y="city", orientation="h", title="(a) Số job Senior+ (tuyệt đối)",
                          labels={"n_jobs_in_level": "Số job", "city": "Thành phố"}, height=bar_h(len(a))))
    b = with_err(ok.sort_values("pct_of_city_jobs", ascending=False).head(topn).sort_values("pct_of_city_jobs"), "pct_of_city_jobs")
    r.plotly_chart(px.bar(b, x="pct_of_city_jobs", y="city", orientation="h", error_x="err_plus", error_x_minus="err_minus",
                          hover_data=["n_jobs_city_total"], title="(b) % job của thành phố là Senior+ (tương đối)",
                          labels={"pct_of_city_jobs": "% job", "city": "Thành phố"}, height=bar_h(len(b))))
    cities = a.city.tolist()
    lv = d[d.city.isin(cities) & (d.level != SENIOR_PLUS)]
    st.plotly_chart(px.bar(lv, x="city", y="pct_of_city_jobs", color="level", barmode="group",
                           category_orders={"city": cities[::-1]},
                           labels={"pct_of_city_jobs": "% job của thành phố", "city": "Thành phố", "level": "Bậc"},
                           title="Cơ cấu từng bậc (Senior, Lead/Principal, Manager+) theo thành phố"))
    small = g[~g.reliable].sort_values("n_jobs_city_total", ascending=False)
    if len(small):
        with st.expander(f"{len(small)} thành phố dưới ngưỡng mẫu (không dùng để kết luận)"):
            st.dataframe(small[["city", "n_jobs_city_total", "n_jobs_in_level", "pct_of_city_jobs"]], hide_index=True)


# ----------------------------------------------------------------------------- khung chính + điều hướng
PAGES = {"Tổng quan": tab_overview, "RQ1 Core stack": tab_rq1, "RQ2 Mật độ": tab_rq2,
         "RQ3 Level": tab_rq3, "RQ4 Địa lý": tab_rq4}
st.sidebar.title("Điều hướng")
page = st.sidebar.radio("Chọn trang", list(PAGES))
st.header(page)
try:
    PAGES[page]()
except Exception as e:  # một trang lỗi không làm trắng cả app
    st.error(f"Không vẽ được trang này: {type(e).__name__}: {e}")
    st.exception(e)