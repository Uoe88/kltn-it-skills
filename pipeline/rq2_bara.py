"""
rq2_bara.py  -  Câu hỏi nghiên cứu 2

"Mật độ xuất hiện của các yêu cầu về Cloud/DevOps, AI/LLM và ngoại ngữ (tiếng Anh, tiếng Pháp, tiếng Nhật)
 trong các bài đăng tuyển dụng biến thiên như thế nào giữa các nhóm nghề?"

Mật độ của một nhóm công nghệ trong một nhóm nghề = % JD của nhóm nghề có ít nhất một skill thuộc nhóm công nghệ.

THUẬT TOÁN
    Bước 1. Đọc file binary jd_skill_binary.csv.
    Bước 2. Xác định các nhóm nghề cần tính (mặc định: tất cả, sắp theo số JD giảm dần).
    Bước 3. Ánh xạ cột skill_* sang nhóm công nghệ bằng cột group của skill_taxonomy.csv:
              Cloud/DevOps = group 'Cloud' + 'DevOps & Infra';   AI/LLM = group 'AI/ML'.
    Bước 4. Mật độ Cloud/DevOps và AI/LLM theo nhóm nghề:
              mẫu    = JD của nhóm nghề có has_skills = 1 và skills_status = has_hard
              %      = số JD có >= 1 skill của nhóm / số JD của mẫu
              CI 95% = Wilson; cờ small_sample khi mẫu < --min-n
    Bước 5. Mật độ ngoại ngữ theo nhóm nghề:
              mẫu    = tất cả JD của nhóm nghề
              chỉ số = % JD yêu cầu tiếng Anh / Nhật / Pháp và 'bất kỳ ngoại ngữ' (OR ba cột)
    Bước 6. Ghi rq2_density.xlsx gồm 3 sheet: Cloud_DevOps, AI_LLM, Ngoai_ngu.

Chạy: python rq2_bara.py --input jd_skill_binary.csv --taxonomy skill_taxonomy.csv --outdir rq2_out
"""
import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

TECH_GROUPS = {
    "Cloud/DevOps": ["Cloud", "DevOps & Infra"],
    "AI/LLM": ["AI/ML"],
}
LANGUAGES = ["english", "japanese", "french"]
LANG_LABEL = {"english": "English", "japanese": "Japanese", "french": "French"}
ANY_LANG = "Bất kỳ ngoại ngữ (gộp)"
ALL_ROLES = "TẤT CẢ nhóm nghề"
KNOWN_NON_TAXONOMY = {f"skill_{l}" for l in LANGUAGES} | {"skill_cert_security"}


def wilson_ci(k, n, z=1.96):
    """Khoảng tin cậy Wilson 95% cho tỷ lệ k/n (0-1); ổn định khi n nhỏ hoặc tỷ lệ sát 0/1."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def slug(canonical):
    """Tên chuẩn trong taxonomy -> tên cột binary (cùng quy tắc với bước tạo binary): 'C++' -> skill_cpp, '.NET' -> skill_dotnet."""
    s = canonical.lower().replace("c++", "cpp").replace("c#", "csharp").replace(".net", "dotnet")
    return "skill_" + re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def make_row(role, scope, label, n, k, n_cols, avg, reliable):
    """Gói một dòng kết quả: % và CI95 của k/n."""
    lo, hi = wilson_ci(k, n)
    return {"role": role, "n_jobs_with_skills": n if scope == "has_hard" else np.nan,
            "n_jobs": n if scope == "all_jobs" else np.nan,
            "tech_group": label, "n_skill_cols_in_group": n_cols,
            "n_jobs_matched": k, "pct": round(100 * k / n, 1) if n else np.nan,
            "ci_low_pct": round(100 * lo, 1) if n else np.nan, "ci_high_pct": round(100 * hi, 1) if n else np.nan,
            "avg_n_skills_in_group": avg, "role_sample_reliable": reliable, "sample": scope}


# BƯỚC 1: ĐỌC DỮ LIỆU
def load_data(path):
    """Đọc file binary; kiểm tra có đủ cột bắt buộc, ô trống của cột skill -> 0."""
    df = pd.read_excel(path) if str(path).endswith((".xlsx", ".xls")) else pd.read_csv(path)
    need = ["expertise_category", "has_skills"] + [f"skill_{l}" for l in LANGUAGES]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"Input thiếu cột: {miss} (cột ngoại ngữ phải là skill_english / skill_japanese / skill_french)")
    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    df[skill_cols] = df[skill_cols].fillna(0).astype(int)
    df["has_skills"] = df["has_skills"].fillna(0).astype(int)
    return df, skill_cols


# BƯỚC 2: CHỌN NHÓM NGHỀ
def pick_roles(df, roles_arg):
    """'ALL' = mọi expertise_category, sắp theo số JD giảm dần; hoặc danh sách người dùng chỉ định."""
    if roles_arg == "ALL":
        return list(df.expertise_category.dropna().value_counts().index)
    roles = [r.strip() for r in roles_arg.split(",") if r.strip()]
    missing = [r for r in roles if r not in df.expertise_category.unique()]
    if missing:
        raise SystemExit(f"Không tìm thấy nhóm nghề: {missing}")
    return roles


# BƯỚC 3: ÁNH XẠ SKILL -> NHÓM CÔNG NGHỆ
def build_tech_groups(skill_cols, taxonomy_path):
    """Tạo {nhóm công nghệ: [cột skill_*]} từ cột group của skill_taxonomy.csv (chỉ skill type = hard)."""
    tax = pd.read_csv(taxonomy_path, encoding="utf-8-sig")
    tax = tax[tax.type == "hard"].copy()
    col_to_group = dict(zip(tax.canonical.map(slug), tax.group))
    groups = {label: [c for c in skill_cols if col_to_group.get(c) in taxo_groups]
              for label, taxo_groups in TECH_GROUPS.items()}
    unmapped = [c for c in skill_cols if c not in col_to_group and c not in KNOWN_NON_TAXONOMY]
    return groups, unmapped


# BƯỚC 4: MẬT ĐỘ CLOUD/DEVOPS VÀ AI/LLM
def tech_density(df, roles, groups, min_n):
    """Với mỗi nhóm nghề và mỗi nhóm công nghệ: % JD có >= 1 skill của nhóm."""
    hard = df[df.has_skills == 1]
    if "skills_status" in hard.columns:
        hard = hard[hard.skills_status == "has_hard"]
    rows = []
    for role in roles + [ALL_ROLES]:
        g = hard if role == ALL_ROLES else hard[hard.expertise_category == role]
        n = len(g)
        for label, cols in groups.items():
            if not cols:
                continue
            has_any = g[cols].any(axis=1)                      # JD có ít nhất 1 skill của nhóm?
            avg = round(g[cols].sum(axis=1).mean(), 1) if n else np.nan   # độ sâu: số skill trung bình trong nhóm
            rows.append(make_row(role, "has_hard", label, n, int(has_any.sum()), len(cols), avg, n >= min_n))
    return pd.DataFrame(rows)


# BƯỚC 5: MẬT ĐỘ NGOẠI NGỮ
def language_density(df, roles, min_n):
    """Với mỗi nhóm nghề: % JD yêu cầu từng ngoại ngữ và 'bất kỳ ngoại ngữ' (OR ba cột)."""
    lang_cols = [f"skill_{l}" for l in LANGUAGES]
    df = df.copy()
    df["_any_lang"] = df[lang_cols].any(axis=1)
    targets = {"_any_lang": ANY_LANG, **{f"skill_{l}": LANG_LABEL[l] for l in LANGUAGES}}
    rows = []
    for role in roles + [ALL_ROLES]:
        g = df if role == ALL_ROLES else df[df.expertise_category == role]
        n = len(g)
        for col, label in targets.items():
            rows.append(make_row(role, "all_jobs", f"Soft skill: {label}", n, int(g[col].sum()), 1, np.nan, n >= min_n))
    return pd.DataFrame(rows)


# BƯỚC 6: HAI SHEET
SHEETS = {
    "Cloud_DevOps": ({"Cloud/DevOps": None}, "has_hard"),
    "AI_LLM": ({"AI/LLM": None}, "has_hard"),
    "Ngoai_ngu": ({f"Soft skill: {ANY_LANG}": "Bất kỳ ngoại ngữ (gộp)", "Soft skill: English": "Tiếng Anh",
                   "Soft skill: Japanese": "Tiếng Nhật", "Soft skill: French": "Tiếng Pháp"}, "all_jobs"),
}


def sheet_table(df, labels, basis, order, n_with_skills, n_all, min_n):
    """Một sheet: mỗi dòng (nhóm nghề [x ngôn ngữ]) có đủ n_jobs, n_jobs_with_skills, n_matched, mẫu số thực dùng, pct, CI."""
    has_lang = any(v for v in labels.values())
    rows = []
    for role in order:
        for label, disp in labels.items():
            r = df[(df.role == role) & (df.tech_group == label)]
            if r.empty:
                continue
            r = r.iloc[0]
            denom = int(n_with_skills[role] if basis == "has_hard" else n_all[role])
            row = {"role": role}
            if has_lang:
                row["language"] = disp
            row.update({"n_jobs": int(n_all[role]), "n_jobs_with_skills": int(n_with_skills[role]),
                        "n_matched": int(r.n_jobs_matched), "n_denominator": denom,
                        "pct_basis": "JD có hard skill" if basis == "has_hard" else "mọi JD của nhóm nghề",
                        "pct": r.pct, "ci_low_pct": r.ci_low_pct, "ci_high_pct": r.ci_high_pct,
                        "small_sample": denom < min_n})
            rows.append(row)
    return pd.DataFrame(rows)


def write_xlsx(path, sheets):
    """Ghi các sheet, định dạng gọn: tiêu đề đậm, đóng băng hàng/cột đầu, % một chữ số thập phân."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for c in ws[1]:
                c.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
                c.fill = PatternFill("solid", fgColor="1F3864")
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws.row_dimensions[1].height = 32
            for i, col in enumerate(d.columns, 1):
                ws.column_dimensions[get_column_letter(i)].width = 22 if col in ("role", "language", "pct_basis") else 14
                for r in range(2, len(d) + 2):
                    cell = ws.cell(row=r, column=i)
                    cell.font = Font(name="Arial", size=10)
                    if col in ("pct", "ci_low_pct", "ci_high_pct"):
                        cell.number_format = "0.0"
            ws.freeze_panes = "B2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="jd_skill_binary.csv")
    ap.add_argument("--taxonomy", default="skill_taxonomy.csv")
    ap.add_argument("--outdir", default="rq2_out")
    ap.add_argument("--roles", default="ALL", help="'ALL' (mặc định: mọi nhóm nghề) hoặc danh sách cách nhau bằng dấu phẩy")
    ap.add_argument("--min-n", type=int, default=50, help="nhóm nghề có ít JD hơn mức này bị gắn cờ cỡ mẫu nhỏ")
    ap.add_argument("--with-total", action="store_true", help="thêm hàng tổng 'TẤT CẢ nhóm nghề' để so mặt bằng chung")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Bước 1-2: đọc dữ liệu, chọn nhóm nghề (mặc định tất cả)
    df, skill_cols = load_data(a.input)
    roles = pick_roles(df, a.roles)

    # Bước 3: ánh xạ skill -> Cloud/DevOps, AI/LLM bằng taxonomy
    groups, unmapped = build_tech_groups(skill_cols, a.taxonomy)

    # Bước 4-5: tính mật độ công nghệ và ngoại ngữ
    tech = tech_density(df, roles, groups, a.min_n)
    lang = language_density(df, roles, a.min_n)
    res = pd.concat([tech, lang], ignore_index=True)

    # Bước 6: một file Excel, ba sheet cùng cấu trúc cột
    order = roles + ([ALL_ROLES] if a.with_total else [])
    n_with_skills = tech.drop_duplicates("role").set_index("role").n_jobs_with_skills
    n_all = lang.drop_duplicates("role").set_index("role").n_jobs
    sheets = {}
    for name, (labels, basis) in SHEETS.items():
        src = tech if basis == "has_hard" else lang
        sheets[name] = sheet_table(src, labels, basis, order, n_with_skills, n_all, a.min_n)
    write_xlsx(out / "rq2_density.xlsx", sheets)
    print(f"Cloud/DevOps = {len(groups['Cloud/DevOps'])} skill; AI/LLM = {len(groups['AI/LLM'])} skill (taxonomy). KHÔNG phân tích theo nguồn.")
    if unmapped:
        print(f"[!] {len(unmapped)} cột skill_* không khớp taxonomy -- cần kiểm tra phiên bản: " + ", ".join(unmapped[:15]))
    for name, d in sheets.items():
        print(f"\n[Sheet {name}]")
        print(d.drop(columns=["pct_basis", "ci_low_pct", "ci_high_pct"]).round(1).to_string(index=False, max_rows=16))
    print(f"\n-> {out}/rq2_density.xlsx")


if __name__ == "__main__":
    main()
