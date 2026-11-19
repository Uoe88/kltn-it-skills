"""fix_job_id.py: đo mức trùng job_id và sinh khóa mới từ job_url.
Chạy: python fixed_job_id.py jobs_raw.xlsx jobs_raw_fixed.xlsx
"""
import hashlib
import sys
from urllib.parse import urlsplit
import argparse
import pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--input", default=r"D:\KLTN\kltn-it-skills\data_private\jobs_raw.xlsx")
ap.add_argument("--output", default=r"D:\KLTN\kltn-it-skills\data_private\jobs_raw_fixed.xlsx")
a = ap.parse_args()
src, dst = a.input, a.output
df = pd.read_excel(src) if src.endswith((".xlsx", ".xls")) else pd.read_csv(src)

# --- 1. Đo mức độ trùng ---
c = df.dropna(subset=["job_id"]).groupby("job_id")["job_url"].nunique()
print(f"{(c > 1).sum()} job_id trỏ tới nhiều URL khác nhau "
      f"({int(c[c > 1].sum())} URL bị ảnh hưởng)")
print("dòng thiếu job_url:", int(df["job_url"].isna().sum()))

# --- 2. Sinh khóa mới từ URL ---
def canon_url(u):
    p = urlsplit(str(u).strip())
    return f"{p.netloc.lower()}{p.path.rstrip('/')}"   # bỏ query/tracking

has = df["job_url"].notna()          # tránh hash chung cho các dòng không có URL
key = df.loc[has, "job_url"].map(
    lambda u: hashlib.sha1(canon_url(u).encode()).hexdigest()[:16])
df["job_id_legacy"] = df["job_id"]
df.loc[has, "job_id"] = df.loc[has, "source_id"].astype(str) + "_" + key

# --- 3. Kiểm tra sau khi sửa ---
d = df.loc[has].duplicated("job_id", keep=False)
print(f"{int(d.sum())} dòng có job_id mới trùng nhau (cùng URL, có thể do crawl lại)")

if dst.endswith((".xlsx", ".xls")):
    df.to_excel(dst, index=False)
else:
    df.to_csv(dst, index=False, encoding="utf-8-sig")
