"""
rq2_bara.py - Câu hỏi nghiên cứu 2 (bản đơn giản)

"Mật độ xuất hiện của các yêu cầu về Cloud/DevOps, AI/GenAI và ngoại ngữ trong các bài đăng
 tuyển dụng biến thiên như thế nào giữa các nhóm nghề?"

Mật độ của một nhóm kỹ năng trong một nhóm nghề = % JD của nhóm nghề có ÍT NHẤT MỘT kỹ năng thuộc nhóm.

Mỗi nhóm kỹ năng cho ra MỘT bảng (một sheet) với cột:
    role | n_jobs | n_jobs_has_skill | pct | ci_low_pct | ci_high_pct | small_sample
      n_jobs           = mẫu số (số JD của nhóm nghề được dùng để tính)
      n_jobs_has_skill = số JD có >= 1 kỹ năng của nhóm
      pct              = n_jobs_has_skill / n_jobs * 100
      ci_low/high_pct  = khoảng tin cậy 95% Wilson
      small_sample     = True nếu n_jobs < --min-n

Mẫu số:
    - Cloud/DevOps, AI/GenAI: chỉ tính JD có trích được hard skill (has_skills == 1),
      vì JD không có skill nào sẽ làm mật độ bị thấp giả tạo.
    - Ngoại ngữ: tính trên TẤT CẢ JD của nhóm nghề.

Tùy chọn:
    --with-total  thêm dòng "ALL" (gộp mọi nhóm nghề) vào mỗi bảng, cùng quy tắc mẫu số.
    --min-n       n_jobs < min-n thì đánh dấu small_sample = True (mặc định 30).

Chạy: python pipeline/rq2_bara.py --input data_private/jd_skill_binary.csv --outdir rq2_out --with-total
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROLE_COL = "expertise_category"

# ---- Nhóm kỹ năng ------------------------------------------------------------------------------------------
# Cloud/DevOps (mới nổi): nền tảng cloud + thực hành cloud-native hiện đại
CLOUD_DEVOPS = [
    # Nền tảng cloud
    "skill_cloud_computing", "skill_aws", "skill_azure", "skill_gcp",
    "skill_alibaba_cloud", "skill_serverless",
    # Cloud-native
    "skill_kubernetes", "skill_docker", "skill_helm",
    "skill_terraform", "skill_infrastructure_as_code",
    "skill_argocd", "skill_gitops",
  
    "skill_devops", "skill_devsecops", "skill_sre", "skill_platform_engineering",
    # Observability
    "skill_prometheus", "skill_grafana", "skill_opentelemetry",
]

# AI/GenAI: dùng công cụ AI hỗ trợ công việc (không gồm ML/deep learning)
AI_GENAI = [
    # Khái niệm / kỹ năng dùng AI
    "skill_generative_ai", "skill_prompt_engineering", "skill_agentic_ai",
    "skill_llm",  "skill_ai_tools",
    # Công cụ AI có tên
    "skill_chatgpt", "skill_claude", "skill_gemini",
    "skill_github_copilot", "skill_microsoft_copilot",
    "skill_cursor", "skill_windsurf",
    "skill_perplexity", "skill_midjourney", "skill_notebooklm", "skill_vibe_coding",
]

LANGUAGES = {
    "English": ["skill_english"],
    "Japanese": ["skill_japanese"],

}

# Tech: mẫu số = JD có hard skill; Ngoại ngữ: mẫu số = tất cả JD
GROUPS = {
    "Cloud_DevOps": (CLOUD_DEVOPS, "has_skill_only"),
    "AI_GenAI": (AI_GENAI, "has_skill_only"),
    **{name: (cols, "all") for name, cols in LANGUAGES.items()},
}


def wilson_ci(k, n, z=1.96):
    """Khoảng tin cậy Wilson 95% cho tỷ lệ k/n (kết quả 0-1)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def density_table(df, roles, cols, with_total=False, min_n=30):
    """Bảng mật độ của một nhóm kỹ năng: mỗi dòng một nhóm nghề (+ dòng ALL nếu with_total)."""
    rows = []
    subsets = [(role, df[df[ROLE_COL] == role]) for role in roles]
    if with_total:
        subsets.append(("ALL", df[df[ROLE_COL].isin(roles)]))
    for role, g in subsets:
        n = len(g)
        k = int(g[cols].any(axis=1).sum()) if n else 0
        lo, hi = wilson_ci(k, n)
        rows.append({
            "role": role,
            "n_jobs": n,
            "n_jobs_has_skill": k,
            "pct": round(100 * k / n, 1) if n else np.nan,
            "ci_low_pct": round(100 * lo, 1) if n else np.nan,
            "ci_high_pct": round(100 * hi, 1) if n else np.nan,
            "small_sample": bool(n < min_n),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="jd_binary_skill.csv")
    ap.add_argument("--outdir", default="rq2_out")
    ap.add_argument("--with-total", action="store_true",
                    help="thêm dòng 'ALL' (gộp mọi nhóm nghề) vào mỗi bảng")
    ap.add_argument("--min-n", type=int, default=30,
                    help="n_jobs < min-n thì đánh dấu small_sample = True (mặc định 30)")
    a = ap.parse_args()

    df = pd.read_excel(a.input) if a.input.endswith((".xlsx", ".xls")) else pd.read_csv(a.input)
    if ROLE_COL not in df.columns:
        raise SystemExit(f"Input thiếu cột '{ROLE_COL}'")

    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    df[skill_cols] = df[skill_cols].fillna(0).astype(int)

    # JD có hard skill: dùng cột has_skills nếu có, không thì suy từ ma trận (loại cột ngoại ngữ)
    lang_cols = [c for cols in LANGUAGES.values() for c in cols]
    if "has_skills" in df.columns:
        has_skill = df["has_skills"].fillna(0).astype(int) == 1
    else:
        has_skill = df[[c for c in skill_cols if c not in lang_cols]].any(axis=1)

    roles = list(df[ROLE_COL].dropna().value_counts().index)  # sắp theo số JD giảm dần

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sheets = {}
    for name, (cols, basis) in GROUPS.items():
        present = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[!] {name}: thiếu cột trong input: {missing}")
        if not present:
            print(f"[!] {name}: không có cột nào, bỏ qua")
            continue
        base = df[has_skill] if basis == "has_skill_only" else df
        sheets[name] = density_table(base, roles, present, with_total=a.with_total, min_n=a.min_n)

    with pd.ExcelWriter(out / "rq2_density.xlsx", engine="openpyxl") as xw:
        for name, t in sheets.items():
            t.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.column_dimensions["A"].width = 18
            for col in "BCDEFG":
                ws.column_dimensions[col].width = 16
            ws.freeze_panes = "B2"

    for name, t in sheets.items():
        print(f"\n[{name}]")
        print(t.to_string(index=False))
    print(f"\n-> {out / 'rq2_density.xlsx'}")


if __name__ == "__main__":
    main()