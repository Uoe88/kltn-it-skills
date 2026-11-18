
"""Chạy rq1_analysis, rq2_bara, rq3_analysis, rq4_geography và ghi kết quả vào results/ cho dashboard đọc.

Cách chạy (chỉ còn MỘT file đầu vào, do build_skill_taxonomy.py tạo ra):
    python pipeline/run_all.py --input data_private/jd_skill_binary.csv --taxonomy data_private/skill_taxonomy.csv
Đặt run_all.py cùng thư mục với 4 script rq*. Cần: pandas, numpy, scipy, openpyxl (rq2 ghi .xlsx).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

for stream in (sys.stdout, sys.stderr):  # tránh lỗi mã hoá khi console Windows không phải UTF-8
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REQUIRED = ["job_uid", "source", "expertise_category", "level_group", "exp_group", "city",
            "has_skills", "n_hard_skills", "n_skill_groups", "skills_status"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="jd_skill_binary.csv", help="file do build_skill_taxonomy.py tạo ra")
    ap.add_argument("--taxonomy", default="skill_taxonomy.csv")
    ap.add_argument("--scripts-dir", default=str(Path(__file__).parent))
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    for label, p in (("input", a.input), ("taxonomy", a.taxonomy)):
        if not Path(p).exists():
            sys.exit(f"Không thấy file {label}: {p} (đường dẫn tính từ thư mục đang đứng, hãy kiểm tra lại).")

    cols = pd.read_csv(a.input, nrows=0).columns
    miss = [c for c in REQUIRED if c not in cols]
    if miss or not any(c.startswith("skill_") for c in cols):
        sys.exit(f"{a.input} thiếu cột {miss} hoặc không có cột skill_* -- hãy chạy lại build_skill_taxonomy.py bản mới "
                 f"(file này phải là jd_skill_binary.csv, không phải job_skill_binary.csv).")

    sd, res = Path(a.scripts_dir), Path(a.results)
    jobs = [("rq1_analysis.py", "rq1_out", ["--report", "--extras"]),
            ("rq2_bara.py", "rq2_out", ["--with-total"]),
            ("rq3_analysis.py", "rq3_out", ["--taxonomy", a.taxonomy]),
            ("rq4_geography.py", "rq4_out", [])]
    for script, _, _ in jobs:
        if not (sd / script).exists():
            sys.exit(f"Không thấy {sd / script} -- đặt run_all.py cùng thư mục với các script rq*.")


    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8"
    }

    # Tạo thư mục results nếu chưa tồn tại
    res.mkdir(parents=True, exist_ok=True)

    for script, out, extra in jobs:

        outdir = res / out
        outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(sd / script),
            "--input", str(Path(a.input).resolve()),
            "--outdir", str(outdir.resolve()),
            *extra
        ]

        print(f"\n{'=' * 60}")
        print(f"ĐANG CHẠY: {script}")
        print(f"{'=' * 60}")
        print(">>", " ".join(cmd))

        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )

        # Hiển thị log của script con
        if r.stdout:
            print(r.stdout)

        if r.stderr:
            print(r.stderr)

        # Dừng nếu script lỗi
        if r.returncode != 0:
            sys.exit(
                f"\n[FAILED] {script} thất bại "
                f"(returncode={r.returncode})"
            )

        # Kiểm tra thư mục kết quả có file được tạo hay không
        output_files = [
            p for p in outdir.rglob("*")
            if p.is_file()
        ]

        if not output_files:
            sys.exit(
                f"\n[WARNING] {script} chạy xong "
                f"nhưng không tìm thấy file trong {outdir}"
            )

        print(f"[OK] {script}: tạo {len(output_files)} file")

    print("\n[SUCCESS] Tất cả RQ đã chạy hoàn thành.")
    print("Mở dashboard: streamlit run app.py")



if __name__ == "__main__":   # import run_all (vd từ app.py/analytics.py) sẽ KHÔNG tự chạy lại rq1..rq4
    main()
