"""
rq4_geography.py  -  Câu hỏi nghiên cứu 4

"Các vị trí cấp Senior/Lead/Manager (Senior/Lead+) chủ yếu tập trung ở thành phố nào?"

THUẬT TOÁN
  1. Loại job thiếu level hoặc thiếu city, và job có city không phải thành phố (NON_CITY_LABELS: Không rõ, Remote).
  2. Senior+ = Senior +  Lead+ (tương ứng với (Senior + Lead/Principal + Manager+, tức level_group = Senior hoặc Lead+). Với mỗi thành phố đếm job từng bậc và Senior+.
  3. Hai cách đọc "tập trung":
         (a) Tỉ lệ số jobs senior+ của mỗi thành phố so với tổng senior+ của các thành phố
         (b) Tỉ lệ số jobs senior+ của mỗi thành phố so với tổng số jobs của thành phố đó
  4. Thành phố có ít hơn --min-n job (đã biết level) bị gắn cờ mẫu nhỏ, vẫn nằm trong file kết quả.
  5. Output rq4_city_level.csv.

Chạy: python rq4_geography.py --input jd_skill_binary.csv --outdir rq4_out
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Dùng cột level_group: Senior = Senior; Lead+ (Lead/Principal + Manager+ (đã gộp sẵn khi làm sạch dữ liệu))
LEVELS_OF_INTEREST = ["Senior", "Lead+"]
# Viết thường hết vì phía so sánh (dòng lọc bên dưới) đã đổi city sang chữ thường
NON_CITY_LABELS = {"không rõ", "remote"}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="jd_skill_binary.xlsx")
    ap.add_argument("--outdir", default="rq4_out")
    ap.add_argument("--min-n", type=int, default=30, help="thành phố cần ít nhất bấy nhiêu job (đã biết level) mới vào bảng xếp hạng chính")
    ap.add_argument("--top-n", type=int, default=10, help="số thành phố hiển thị trong report")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(a.input) if a.input.endswith((".xlsx", ".xls")) else pd.read_csv(a.input)
    is_non_city = df.city.astype(str).str.strip().str.lower().isin(NON_CITY_LABELS)
    known = df[df.level_group.notna() & df.city.notna() & ~is_non_city].copy()
    n_excluded = len(df) - len(known)
    n_non_city = int(is_non_city.sum())
    is_senior_plus = known.level_group.isin(LEVELS_OF_INTEREST)
    total_senior_plus = int(is_senior_plus.sum())

    city_n = known.groupby("city").size().rename("n_jobs_city_total")
    rows = []
    for city, g in known.groupby("city"):
        n_city = len(g)
        for level in LEVELS_OF_INTEREST + ["Senior+ (gộp 2 bậc trên)"]:
            k = int((g.level_group == level).sum()) if level != "Senior+ (gộp 2 bậc trên)" \
                else int(g.level_group.isin(LEVELS_OF_INTEREST).sum())
            pct_city = 100 * k / n_city if n_city else np.nan
            pct_of_all_senior = 100 * k / total_senior_plus if level == "Senior+ (gộp 2 bậc trên)" and total_senior_plus else np.nan
            lo, hi = wilson_ci(k, n_city)
            rows.append({"city": city, "level": level, "n_jobs_in_level": k, "n_jobs_city_total": n_city,
                         "pct_of_city_jobs": round(pct_city, 1), "ci_low_pct": round(100 * lo, 1), "ci_high_pct": round(100 * hi, 1),
                         "pct_of_all_senior_plus_jobs": round(pct_of_all_senior, 1) if not np.isnan(pct_of_all_senior) else np.nan,
                         "city_sample_reliable": n_city >= a.min_n})
    table = pd.DataFrame(rows).sort_values(["level", "n_jobs_in_level"], ascending=[True, False])
    table.to_csv(out / "rq4_city_level.csv", index=False, encoding="utf-8-sig")

    rep = [f"Job có cả city và level xác định: {len(known)}/{len(df)} ({n_excluded} job bị loại vì thiếu city hoặc level)",
           f"Tổng job Senior+ (Senior + Lead/Principal + Manager+, tức level_group = Senior hoặc Lead+): {total_senior_plus}",
           f"Ngưỡng cỡ mẫu thành phố: >= {a.min_n} job (đã biết level) mới vào bảng xếp hạng chính",
           f"(Đã loại {n_non_city} job có city là 'Không rõ' hoặc 'Remote' khỏi mọi bảng: đây không phải thành phố thật)\n"]

    main_table = table[(table.level == "Senior+ (gộp 2 bậc trên)") & table.city_sample_reliable]
    small_cities = table[(table.level == "Senior+ (gộp 2 bậc trên)") & ~table.city_sample_reliable].sort_values("n_jobs_city_total", ascending=False)

    rep.append("=== (a) TUYỆT ĐỐI -- thành phố có NHIỀU job Senior+ nhất (số lượng thô) ===")
    top_abs = main_table.sort_values("n_jobs_in_level", ascending=False).head(a.top_n)
    for r in top_abs.itertuples():
        rep.append(f"  {r.city:<16} {r.n_jobs_in_level:>4} job Senior+  "
                   f"({r.pct_of_all_senior_plus_jobs:>4.1f}% tổng số job Senior+ toàn dữ liệu)")

    rep.append("\n=== (b) TƯƠNG ĐỐI -- thành phố có TỶ LỆ job Senior+ cao nhất (trong số job của thành phố đó) ===")
    top_rel = main_table.sort_values("pct_of_city_jobs", ascending=False).head(a.top_n)
    for r in top_rel.itertuples():
        rep.append(f"  {r.city:<16} {r.pct_of_city_jobs:>5.1f}% job là Senior+  "
                   f"(CI95 {r.ci_low_pct:.1f}-{r.ci_high_pct:.1f}%, n={r.n_jobs_city_total} job tại thành phố này)")

    rep.append("\n=== Phân theo từng bậc (Senior / Lead+), chỉ thành phố đủ mẫu ===")
    piv = table[table.city.isin(main_table.city) & (table.level != "Senior+ (gộp 2 bậc trên)")]
    piv_pct = piv.pivot(index="city", columns="level", values="pct_of_city_jobs")
    piv_n = piv.pivot(index="city", columns="level", values="n_jobs_in_level")
    piv_pct.insert(0, "n_jobs_city_total", city_n.reindex(piv_pct.index))
    rep.append("  % trong tổng job của thành phố:")
    rep.append("  " + piv_pct.sort_values("n_jobs_city_total", ascending=False).to_string().replace("\n", "\n  "))
    rep.append("\n  Số job tuyệt đối:")
    rep.append("  " + piv_n.loc[piv_pct.sort_values('n_jobs_city_total', ascending=False).index].to_string().replace("\n", "\n  "))

    if len(small_cities):
        rep.append(f"\n=== Thành phố dưới ngưỡng mẫu (n < {a.min_n}), KHÔNG đưa vào xếp hạng trên nhưng vẫn có trong rq4_city_level.csv ===")
        for r in small_cities.itertuples():
            rep.append(f"  {r.city:<16} n={r.n_jobs_city_total:>3} job, trong đó {r.n_jobs_in_level} Senior+ ({r.pct_of_city_jobs:.0f}% -- cỡ mẫu quá nhỏ để diễn giải)")

    rep.append(f"\n{n_excluded} job bị loại khỏi toàn bộ phân tích này vì thiếu level xác định, thiếu city, hoặc city là "
               f"'Không rõ'/'Remote' -- không gán đại khi thiếu dữ liệu.")

    print("\n".join(rep))
    print(f"\n-> {out}/rq4_city_level.csv")


if __name__ == "__main__":
    main()