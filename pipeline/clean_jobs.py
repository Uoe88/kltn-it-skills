"""
clean_jobs.py  làm sạch jobs_raw (ITviec + LinkedIn + Xóm Jobs + TopCV)

Chạy:
    python clean_jobs.py --input jobs_raw.xlsx --outdir clean_out --as-of 2026-09-20

Thứ tự xử lý:
    0. Tạo cột `source` từ source_id (báo lỗi nếu source_id lạ), điền job_id trống
       (đầu vào giả định đã sửa lệch cột Xóm Jobs)
    1. Location -> city (tên thành phố/tỉnh chuẩn; địa chỉ dài thì tìm tên thành phố ở vị trí phải nhất)
    2. Khử trùng lặp theo (title, company, city) đã chuẩn hóa, gộp field thiếu; xuất dedupe_review.csv (cặp nghi trùng
    3. Parse số năm kinh nghiệm  (experience_raw -> exp_min / exp_max)
    4. Chuẩn hóa level-> Phân tích level_lo, level_hi -> level group (từ nhãn `level`, hỗ trợ khoảng như "Middle | Senior")
    5. Cờ mâu thuẫn level <-> exp -> xuất jobs_clean.csv + level_map.csv + dedupe_review.csv

"""
import argparse
import csv  
import hashlib
import itertools  
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd



# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
#  Các cột text cần làm sạch chuỗi thoát (mô tả/yêu cầu của Xóm Jobs hay chứa \" )
TEXT_COLS = ["job_title", "company_name", "location", "salary_raw", "experience_raw",
             "description_raw", "requirements_raw", "skills_raw", "expertise_category"]

#Làm sạch chuỗi văn bản và loại bỏ cột không cần thiết
def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # bỏ cột rác "Unnamed: n" (xuất hiện nếu file đầu vào từng bị Excel mở và lưu lại)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    for c in TEXT_COLS:
        if c in df.columns:
            s = df[c].astype("string")                       # kiểu string để giữ nguyên giá trị trống (NA)
            s = s.str.replace('\\"', '"', regex=False)       # chuỗi thoát JSON  \"  ->  "
            s = s.str.replace("\r\n", "\n", regex=False)     # thống nhất kiểu xuống dòng
            df[c] = s
    return df

#Loại bỏ dấu tiếng Việt và chuyển chuỗi về dạng ASCII
def strip_accents(s) -> str:  # bỏ dấu, ký tự lạ -> ascii
    s = str(s).replace("đ", "d").replace("Đ", "D")          # 'đ' không tách được bằng NFKD
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

# 	Chuẩn hóa chuỗi bằng cách bỏ dấu, chuyển chữ thường và loại bỏ ký tự đặc biệt.
def plain(s, keep_symbols="+#") -> str:
    """Chuẩn hóa chuỗi: bỏ dấu, chữ thường, ký tự lạ -> khoảng trắng (giữ + và # cho C++, C#)."""
    s = strip_accents(s).lower()
    allowed = re.escape(keep_symbols)
    pattern = rf"[^a-z0-9{allowed}]+"

    s = re.sub(pattern, " ", s)
    return re.sub(r"\s+", " ", s).strip()





# --------------------------------------------------------------------------
# 0. Nguồn + job_id
# --------------------------------------------------------------------------
SOURCE_LABEL = {"ITV": "ITviec", "LNK": "LinkedIn", "topcv": "TopCV", "xjob": "XomJobs"}  # Chuẩn hóa tên nguồn
SOURCE_PRIORITY = {"ITviec": 0, "LinkedIn": 1, "TopCV": 2, "XomJobs": 3}   # nguồn gốc ưu tiên hơn nguồn tổng hợp	
# Xác định nguồn thu thập dữ liệu và bổ sung mã định danh công việc còn thiếu.
def add_source(df: pd.DataFrame):
    df = df.copy()
    # dấu hiệu dữ liệu Xóm Jobs bị lệch cột: job_title chứa đúng chữ "xjob"
    bad = df["job_title"].astype(str).str.strip().str.lower().eq("xjob")
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} Hãy kiểm tra lại dữ liệu đầu vào có thể lệch")
    df["source"] = df["source_id"].map(SOURCE_LABEL)
    # source_id lạ (URL, tên thành phố... do vỡ cột) -> dừng, không để lọt xuống bước sau
    if df["source"].isna().any():
        raise SystemExit(f"{int(df['source'].isna().sum())} dòng có source_id không hợp lệ: "
                         f"{df.loc[df['source'].isna(), 'source_id'].unique()[:5]}")
    df["job_id"] = df["job_id"].fillna(df["source_id"])
    return df


# --------------------------------------------------------------------------
# 1. Location -> city
# --------------------------------------------------------------------------
# ạo các biểu thức chính quy để nhận diện tên tỉnh, thành phố.
CITY_ALIASES = {
    "Hà Nội": ["hanoi"],
    "Hồ Chí Minh": ["hcmc", "tp hcm", "tphcm", "hcm", "sai gon", "saigon"],
    "Đà Nẵng": ["danang"], "Cần Thơ": [], "Hải Phòng": [], "Hải Dương": [], "Nghệ An": [],
    "Huế": ["hue", "thua thien hue"], "Bà Rịa - Vũng Tàu": ["ba ria", "vung tau"], "Đắk Lắk": ["dak lak", "daklak"],
}
OTHER_PROVINCES = [
    "An Giang", "Bạc Liêu", "Bắc Giang", "Bắc Kạn", "Bắc Ninh", "Bến Tre", "Bình Dương", "Bình Định", "Bình Phước",
    "Bình Thuận", "Cà Mau", "Cao Bằng", "Đắk Nông", "Điện Biên", "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam",
    "Hà Tĩnh", "Hậu Giang", "Hòa Bình", "Hưng Yên", "Khánh Hòa", "Kiên Giang", "Kon Tum", "Lai Châu", "Lâm Đồng", "Lạng Sơn",
    "Lào Cai", "Long An", "Nam Định", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên", "Quảng Bình", "Quảng Nam", "Quảng Ngãi",
    "Quảng Ninh", "Quảng Trị", "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình", "Thái Nguyên", "Thanh Hóa", "Tiền Giang",
    "Trà Vinh", "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái",
]
OVERSEAS = "Nước ngoài"
OVERSEAS_RE = re.compile(r"\b(overseas|france|paris|singapore|usa|united states|japan|tokyo|germany|australia|"
                         r"thailand|malaysia|philippines|korea|india|canada|uk)\b")

# Trích xuất và chuẩn hóa tên tỉnh, thành phố từ thông tin địa điểm hoặc tiêu đề công việc.
def _city_regexes():
    """Tạo regex cho từng tỉnh/thành (tên chuẩn không dấu + alias), ưu tiên tên dài trước."""
    rx = {}
    for label, alts in {**CITY_ALIASES, **{p: [] for p in OTHER_PROVINCES}}.items():
        names = {plain(label)} | set(alts)
        rx[label] = re.compile(r"\b(" + "|".join(sorted((re.escape(n) for n in names), key=len, reverse=True)) + r")\b")
    return rx


CITY_RX = _city_regexes()
TITLE_CITY_TAGS = [
    ("Hà Nội", r"\b(hn|ha noi|hanoi)\b"),
    ("Hồ Chí Minh", r"\b(hcm|hcmc|tphcm|ho chi minh|sai gon|saigon)\b"),
    ("Đà Nẵng", r"\b(da nang|danang)\b"),
]

# Kiểm tra giá trị có bị thiếu hoặc rỗng hay không.
def parse_city(loc, title):
    """
    Tìm tên thành phố/tỉnh trong toàn bộ chuỗi location, chọn cái xuất hiện Ở VỊ TRÍ PHẢI NHẤT và dán nhãn
    """
    if pd.notna(loc) and str(loc).strip():
        t = plain(loc)
        best, best_end = None, -1
        for label, rx in CITY_RX.items():
            for m in rx.finditer(t):
                if m.end() > best_end:        # lấy match nằm phải nhất (thành phố thường ở cuối địa chỉ)
                    best, best_end = label, m.end()
        if best:
            return best, "location"
        if OVERSEAS_RE.search(t):
            return OVERSEAS, "location"
    # không thấy tên thành phố (vd chỉ 'Vietnam') -> thử tag trong title: [HN], (HCM), 'Tại Hà Nội'
    tt = plain(title)
    for city, pat in TITLE_CITY_TAGS:
        if re.search(pat, tt):
            return city, "title"
    return "Không rõ", "missing"


# --------------------------------------------------------------------------
# 2. Dedupe
# --------------------------------------------------------------------------
LEGAL_RE = re.compile(r"\b(cong ty|tnhh|co phan|cp|jsc|joint stock company|company limited|co|ltd|llc|inc|"
                      r"corp|corporation|pte|pty|vietnam|viet nam|vn|group|holdings?)\b")
LOC_TAG_RE = re.compile(r"[\[\(]\s*(hn|hcm|hcmc|ha noi|hanoi|ho chi minh|da nang|remote|hybrid|onsite)\s*[\]\)]", re.I)
SALARY_IN_TITLE_RE = re.compile(r"\b(offer\s+)?(up ?to|upto)\s*\$?[\d.,]+\s*[kmb$]*\w*", re.I)

#Kiểm tra giá trị có bị thiếu hoặc rỗng hay không.
def _blank(x) -> bool:
    """True nếu giá trị trống (NaN/None/chuỗi chỉ có khoảng trắng)."""
    return pd.isna(x) or not str(x).strip()

# Chuẩn hóa tiêu đề công việc để phục vụ so sánh và khử trùng lặp.
def title_key(t):
    """Khóa so khớp title: bỏ tag địa điểm [HN], lương 'up to ...' rồi chuẩn hóa. Title trống -> ''."""
    if _blank(t):
        return ""
    t = SALARY_IN_TITLE_RE.sub(" ", LOC_TAG_RE.sub(" ", str(t)))
    return plain(t)

# Chuẩn hóa tên công ty bằng cách loại bỏ hậu tố pháp lý và ký tự không cần thiết.
def company_key(c):
    """Khóa so khớp công ty: bỏ hậu tố pháp lý (TNHH, JSC, ...) rồi chuẩn hóa. Công ty trống -> ''."""
    if _blank(c):
        return ""
    return re.sub(r"\s+", " ", LEGAL_RE.sub(" ", re.sub(r"[.,]", " ", plain(c)))).strip()


# Các cột sẽ được "gộp" (lấy giá trị không trống đầu tiên trong nhóm trùng) nếu bản được chọn bị thiếu
COALESCE_COLS = ["level", "salary_raw", "experience_raw", "deadline", "description_raw",
                 "requirements_raw", "skills_raw", "expertise_category", "location"]
FUZZY_RATIO = 0.90   # ngưỡng giống nhau của title để đưa vào dedupe_review.csv
UNKNOWN_CITY = "Không rõ"

#Khử trùng lặp tin tuyển dụng, hợp nhất thông tin và phát hiện các cặp tin nghi trùng.
def dedupe(df: pd.DataFrame):
    """
    Khử trùng lặp theo (title, công ty, city) đã chuẩn hóa. Trả (df đã gộp, review cặp nghi trùng).
    """
    df = df.reset_index(drop=True)
    df["_row_order"] = range(len(df))                 # dùng làm tie-break cuối cùng khi sắp xếp
    df["_tk"] = df["job_title"].map(title_key)        # khóa title
    df["_ck"] = df["company_name"].map(company_key)   # khóa công ty

    # (a) khóa rỗng -> khóa riêng cho từng tin (dựa trên URL, ổn định giữa các lần chạy)
    #     để các tin thiếu title/công ty không bị gộp chung thành một
    ident = df["job_url"].where(df["job_url"].notna(), pd.Series("row" + df.index.astype(str), index=df.index))
    for col, tag in (("_tk", "__notitle_"), ("_ck", "__nocompany_")):
        m = df[col].eq("")
        df.loc[m, col] = tag + ident[m].astype(str)
    anon = df["_tk"].str.startswith("__notitle_") | df["_ck"].str.startswith("__nocompany_")

    # (b) city "Không rõ" -> mượn city nếu (title, công ty) chỉ có đúng 1 city cụ thể ở các tin khác
    df["_city"] = df["city"]
    g_known = df[df["city"].ne(UNKNOWN_CITY) & ~anon].groupby(["_tk", "_ck"])["city"]
    only = g_known.first()[g_known.nunique() == 1].to_dict()       # {(title, công ty): city duy nhất}
    fill = pd.Series(list(zip(df["_tk"], df["_ck"])), index=df.index).map(only)
    infer = df["city"].eq(UNKNOWN_CITY) & ~anon & fill.notna()
    df.loc[infer, "_city"] = fill[infer]
    df["city_inferred"] = infer                                    # cờ: city này được suy luận, không đọc trực tiếp

    df["_grp"] = df.groupby(["_tk", "_ck", "_city"], sort=False).ngroup()   # cùng khóa = cùng 1 tin

    # (c) cặp title gần giống để review tay: so sánh trên title DUY NHẤT trong mỗi (công ty, city), bỏ tin ẩn danh
    review = []
    for (ck, city), g in df[~anon].groupby(["_ck", "_city"]):
        u = g.drop_duplicates("_tk").to_dict("records")
        for x, y in itertools.combinations(u, 2):
            r = SequenceMatcher(None, x["_tk"], y["_tk"]).ratio()
            if r >= FUZZY_RATIO:
                review.append({"company": x["company_name"], "city": city, "ratio": round(r, 3),
                               "title_a": x["job_title"], "url_a": x["job_url"],
                               "title_b": y["job_title"], "url_b": y["job_url"]})
    review = pd.DataFrame(review, columns=["company", "city", "ratio", "title_a", "url_a", "title_b", "url_b"])

    # (d) chọn bản đại diện của mỗi nhóm, tie-break xác định để kết quả giống nhau giữa các lần chạy
    df["_score"] = df[COALESCE_COLS].notna().sum(axis=1)                       # bản đủ thông tin hơn được ưu tiên
    df["_sp"] = df["source"].map(SOURCE_PRIORITY).fillna(len(SOURCE_PRIORITY))  # ưu tiên nguồn gốc hơn nguồn tổng hợp
    df = df.sort_values(["_grp", "_score", "_sp", "crawled_at", "job_url", "_row_order"],
                        ascending=[True, False, True, False, True, True], na_position="last")

    rows = []
    for _, g in df.groupby("_grp", sort=False):
        base = g.iloc[0].copy()                 # bản tốt nhất của nhóm
        filled = []                             # ghi lại field nào được bổ sung từ nguồn nào
        for c in COALESCE_COLS:
            if pd.isna(base[c]):
                nn = g[c].dropna()
                if len(nn):
                    base[c] = nn.iloc[0]
                    filled.append(f"{c}:{g.loc[nn.index[0], 'source']}")
        base["coalesced_from"] = ";".join(filled)

        # cờ mâu thuẫn giữa các bản trùng (bỏ giá trị exp > 20 vì add_experience coi là nhiễu)
        exps = {v for v in (parse_exp(x)[0] for x in g["experience_raw"].dropna()) if pd.notna(v) and v <= 20}
        lvls = {v for v in (levels_in(x)[0] for x in g["level"]) if pd.notna(v)}
        base["merge_conflict"] = ";".join(n for n, v in (("exp", exps), ("level", lvls)) if len(v) > 1)

        base["city"] = base["_city"]
        base["city_inferred"] = bool(g["city_inferred"].any())
        if base["city_inferred"] and base["city_source"] == "missing" and base["city"] != UNKNOWN_CITY:
            base["city_source"] = "inferred"    # city lấy từ tin khác trong nhóm, không phải từ location của bản này
        base["n_duplicates"] = len(g) - 1
        base["sources_merged"] = "|".join(sorted(g["source"].unique(),
                                                 key=lambda s: SOURCE_PRIORITY.get(s, len(SOURCE_PRIORITY))))
       
        
        base["source_job_id"] = "|".join(g["job_id"].astype(str))                          # giữ id gốc để truy vết
        base["source_urls"] = "|".join(g["job_url"].dropna().astype(str).unique())         # giữ URL của mọi bản trùng
        # id ổn định giữa các lần chạy: hash khóa nhóm (xem lưu ý về city suy luận ở đầu file)
        base["job_uid"] = hashlib.md5(f"{base['_tk']}|{base['_ck']}|{base['_city']}".encode()).hexdigest()[:12]
        rows.append(base)
    out = (pd.DataFrame(rows)
             .drop(columns=["_tk", "_ck", "_grp", "_score", "_sp", "_row_order", "_city"])
             .reset_index(drop=True))
    return out, review


# --------------------------------------------------------------------------
# 3. Experience
# --------------------------------------------------------------------------
NUM = r"(\d{1,2})"
RANGE = re.compile(NUM + r"\s*(?:-|–|—|to|đến)\s*" + NUM + r"\s*\+?\s*(?:years?|yrs?|năm)", re.I)   # "2-4 năm"
PLUS = re.compile(r"(?:(trên|over|more than|at least|tối thiểu|ít nhất)\s*)?" + NUM + r"\s*(\+)?\s*(?:years?|yrs?|năm)", re.I)  # "3+ năm"
UNDER = re.compile(r"(dưới|under|less than)\s*" + NUM + r"\s*(?:years?|năm)", re.I)                 # "dưới 1 năm"

# Trích xuất số năm kinh nghiệm tối thiểu và tối đa từ dữ liệu thô.
def parse_exp(s):
    """Trả (exp_min, exp_max) từ chuỗi kinh nghiệm; không đọc được -> (NaN, NaN)."""
    if pd.isna(s):
        return np.nan, np.nan
    t = str(s)[:400]                       # có ô chứa nguyên đoạn JD, chỉ lấy phần đầu
    if re.search(r"không yêu cầu|no experience|không cần", t, re.I):
        return 0, 0
    if m := RANGE.search(t):
        a, b = sorted((int(m[1]), int(m[2])))
        return a, b
    if m := UNDER.search(t):
        return 0, int(m[2])
    if m := PLUS.search(t):
        n = int(m[2])
        return n, (np.nan if (m[1] or m[3]) else n)   # "3+" / "trên 3" -> không có cận trên
    return np.nan, np.nan

# Chuẩn hóa và phân nhóm yêu cầu kinh nghiệm của tin tuyển dụng.
def add_experience(df):
    df = df.copy()
    r = df["experience_raw"].map(parse_exp)
    df["exp_min"] = r.str[0]
    df["exp_max"] = r.str[1]
    df.loc[df["exp_min"] > 20, ["exp_min", "exp_max"]] = np.nan     # '30+ years', '115 NĂM' -> coi là nhiễu
    df["exp_group"] = (pd.cut(df["exp_min"], [-1, 1, 3, 5, 8, np.inf],
                              labels=["0–1", "2–3", "4–5", "6–8", "9+"])
                       .astype("string").fillna("Không rõ"))
    return df


# --------------------------------------------------------------------------
# 4. Level  (bảng ladder = "bảng chuẩn hóa level")
# --------------------------------------------------------------------------
G = ["entry", "middle", "senior", "lead+"]   # 4 nhóm level, chỉ số 0..3

# từ khóa (đã qua chuẩn hóa không dấu) -> (thấp, cao) theo chỉ số nhóm
KEYWORD = {
    "intern": (0, 0), "thuc tap": (0, 0), "fresher": (0, 0), "fresh graduate": (0, 0),
    "junior": (0, 0), "jr": (0, 0), "entry level": (0, 0),
    "middle": (1, 1), "mid level": (1, 1), "intermediate": (1, 1), "associate": (1, 1), "mid": (1, 1),
    "mid senior level": (1, 2),                  # nhãn thô LinkedIn "Mid-Senior level"
    "senior": (2, 2), "sr": (2, 2),
    "lead": (3, 3), "leader": (3, 3), "principal": (3, 3), "staff": (3, 3), "expert": (3, 3),
    "manager": (3, 3), "head": (3, 3), "director": (3, 3), "vp": (3, 3), "chief": (3, 3),
    "cto": (3, 3), "cio": (3, 3), "truong nhom": (3, 3), "truong phong": (3, 3), "giam doc": (3, 3),
}
# regex bắt từ khóa, ưu tiên cụm dài trước (để "mid senior level" thắng "mid" và "senior")
RX = re.compile(r"\b(" + "|".join(map(re.escape, sorted(KEYWORD, key=len, reverse=True))) + r")\b")
SEP = re.compile(r"\s*(?:/|\||,|-|–|&|\bor\b|\bhoac\b)\s*")   # dấu phân cách kiểu "hoặc": Middle | Senior

# 	Nhận diện và xác định khoảng cấp bậc công việc từ nhãn level ban đầu.
def levels_in(text):
    """
    Trả (level_lo, level_hi, level_src) từ nhãn level.
      - Rỗng / không có từ khóa  -> (NaN, NaN, "unknown")
      - Một mức                  -> (x, x, "label")
      - 'Middle | Senior'        -> (1, 2, "label")  (các từ khóa cách nhau bằng dấu '|', '/', ',', '-', 'or'...)
      - Nhiều mức không phải lựa chọn (vd 'Senior Lead') -> lấy mức cao nhất (hi, hi)
    """
    if pd.isna(text):
        return np.nan, np.nan, "unknown"
    # KHÔNG dùng plain() ở đây vì plain() xóa mất dấu | / , - mà SEP cần để nhận ra khoảng level
    t = strip_accents(text).lower().replace("mid-senior", "mid senior")   # nhãn LinkedIn có gạch nối trong từ khóa
    t = re.sub(r"[^a-z0-9+#/|,&\-]+", " ", t)                              # giữ lại các dấu phân cách
    t = re.sub(r"\s+", " ", t).strip()
    hits = [(m.start(), m.end(), KEYWORD[m.group()]) for m in RX.finditer(t)]
    if not hits:
        return np.nan, np.nan, "unknown"
    lo = min(h[2][0] for h in hits)
    hi = max(h[2][1] for h in hits)
    if len(hits) == 1 or lo == hi:
        return lo, hi, "label"
    # các từ khóa liền kề phải cách nhau bằng dấu "hoặc" thì mới coi là khoảng level
    is_choice = all(SEP.fullmatch(t[a[1]:b[0]]) for a, b in zip(hits, hits[1:]))
    return (lo, hi, "label") if is_choice else (hi, hi, "label")

# Chuẩn hóa cấp bậc công việc và phân loại thành các nhóm level.
def add_level(df):
    df = df.copy()
    out = [levels_in(text) for text in df["level"]]                 # list các tuple (lo, hi, src)
    df["level_lo"], df["level_hi"], df["level_src"] = zip(*out)     # hàm trả đủ 3 giá trị nên unpack được
    df["level_group"] = df["level_lo"].map(lambda v: G[int(v)] if pd.notna(v) else None)
    df["is_multi_level"] = (
        df["level_lo"].notna()
        & (df["level_lo"] != df["level_hi"])
    )
    return df


# --------------------------------------------------------------------------
# Main Điều phối toàn bộ quy trình làm sạch, chuẩn hóa, kiểm tra và xuất dữ liệu.
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"D:\KLTN\kltn-it-skills\data_private\jobs_raw_new.xlsx")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--as-of", default=None, help="ngày 'hôm nay' để cờ crawled_at bất thường (YYYY-MM-DD)")
    a = ap.parse_args()
    out = Path(a.outdir) if a.outdir else Path(a.input).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp(a.as_of) if a.as_of else pd.Timestamp.today().normalize()
    rep = []

    df = pd.read_excel(a.input) if a.input.endswith((".xlsx", ".xls")) else pd.read_csv(a.input)
    

    # làm sạch chuỗi thoát + bỏ cột rác ngay sau khi đọc
    df = sanitize(df)
    # ép crawled_at về datetime; giá trị hỏng -> NaT thay vì làm sập các bước so sánh ngày phía sau
    df["crawled_at"] = pd.to_datetime(df["crawled_at"], errors="coerce")

    # 0 Xác định nguồn dữ liệu 
    df = add_source(df)
    # thống kê số lượng tin thô theo nguồn
    n_raw = len(df)
    print("[PHỄU] JD thô:", n_raw)
    print(df["source"].value_counts().to_string())
    
    # 1 Trích xuất và chuẩn hóa thông tin tỉnh/thành phố từ location và job_title
    res = df.apply(lambda r: parse_city(r["location"], r["job_title"]), axis=1, result_type="expand")
    df["city"], df["city_source"] = res[0], res[1]
    rep.append("[1] City (trước dedupe): " + df["city"].value_counts().to_dict().__str__())
    rep.append("    city_source: " + df["city_source"].value_counts().to_dict().__str__())

    # 2 Khử trùng lặp tin tuyển dụng, hợp nhất thông tin và xuất danh sách các cặp nghi trùng
    n0 = len(df)     
    df, review = dedupe(df)
    review.to_csv(out / "dedupe_review.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_NONNUMERIC)
    rep.append(f"[2] Dedupe: {n0} -> {len(df)} dòng (bỏ {n0 - len(df)}); "
               f"dòng có >=1 bản trùng: {(df['n_duplicates'] > 0).sum()}; "
               f"gộp chéo nguồn: {df['sources_merged'].str.contains(r'[|]').sum()}; "
               f"cặp title gần giống cần review tay: {len(review)} (dedupe_review.csv); "
               f"city suy luận: {int(df['city_inferred'].sum())}; nhóm mâu thuẫn exp/level: {int((df['merge_conflict'] != '').sum())}")
    # thống kê số lượng bản trùng cùng nguồn và khác nguồn
    cross = int(df["sources_merged"].str.count(r"\|").sum())   # bản trùng khác nguồn bị gộp
    same = int(df["n_duplicates"].sum()) - cross                # bản trùng cùng nguồn bị gộp
    print(f"[PHỄU] Gộp trùng cùng nguồn : -{same}  -> còn {n_raw - same}")
    print(f"[PHỄU] Gộp trùng khác nguồn : -{cross} -> còn {len(df)}")

    # 3 Trích xuất số năm kinh nghiệm và phân nhóm kinh nghiệm
    df = add_experience(df)
    rep.append(f"[3] Exp: có {df.exp_min.notna().sum()}, thiếu {df.exp_min.isna().sum()}")
    rep.append("    exp_group: " + df["exp_group"].value_counts().to_dict().__str__())

    # 4 Chuẩn hóa cấp bậc công việc và phân loại theo nhóm level
    df = add_level(df)
    rep.append("[4] level_src: " + df["level_src"].value_counts().to_dict().__str__())
    rep.append("    level_group: " + df["level_group"].value_counts(dropna=False).to_dict().__str__())
    rep.append(f"    level nhiều mức: {int(df.is_multi_level.sum())}")

    # cờ mâu thuẫn level ↔ exp: level entry mà đòi >=5 năm, hoặc senior+ mà chỉ đòi <=1 năm
    df["level_exp_conflict"] = ((df["level_hi"] == 0) & (df["exp_min"] >= 5)) | \
                               ((df["level_lo"] >= 2) & (df["exp_min"] <= 1))
    rep.append(f"    mâu thuẫn level↔exp cần rà tay: {int(df['level_exp_conflict'].sum())}")
    # cảnh báo khác
    fut = (df["crawled_at"] > as_of + pd.Timedelta(days=1)).sum()
    rep.append(f"[!] crawled_at: {df.crawled_at.min()} -> {df.crawled_at.max()}; "
               f"{fut} dòng có crawled_at sau {as_of.date()} (kiểm tra đồng hồ máy crawl)")
    rep.append(f"[!] expertise_category trống: {df.expertise_category.isna().sum()}; "
               f"skills_raw trống: {df.skills_raw.isna().sum()}; city 'Không rõ': {(df.city == 'Không rõ').sum()}")
    # 
    print(f"[PHỄU] Sau parse kinh nghiệm / chuẩn hóa level: {len(df)} (không loại dòng)")
    print(f"[PHỄU] JD sạch: {len(df)}  (tổng bị loại: {n_raw - len(df)})")
    assert n_raw - len(df) == same + cross, "phễu không khớp"
    cols = ["job_uid", "source", "job_title", "company_name", "city", "city_source","city_inferred", "n_duplicates", "source_job_id",
            "level_group", "level_lo", "level_hi", "level_exp_conflict",
            "exp_min", "exp_max", "exp_group", 
            "expertise_category", "skills_raw", "requirements_raw", "description_raw",
            "job_url", "updated_at", "deadline", "crawled_at", "sources_merged",
            "coalesced_from", "merge_conflict", "source_urls"] 

    # kiểm tra tính toàn vẹn TRƯỚC khi ghi; sai là dừng ngay, không ghi file hỏng
    assert df["job_uid"].astype(str).str.fullmatch(r"[0-9a-f]{12}").all(), "job_uid hỏng (không phải hash 12 ký tự)"
    assert df["source"].isin(SOURCE_LABEL.values()).all(), "source không hợp lệ"
    assert df["level_group"].dropna().isin(G).all(), "level_group hỏng"
    assert df["exp_group"].isin(["0–1", "2–3", "4–5", "6–8", "9+", "Không rõ"]).all(), "exp_group hỏng"

    # QUOTE_NONNUMERIC: mọi chuỗi được bọc "..." -> dấu ", dấu phẩy, xuống dòng trong text không vỡ cột
    df[cols].to_csv(out / "jobs_clean.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_NONNUMERIC)

    lm = [{"source_label": k, "group_lo": G[lo], "group_hi": G[hi]}
          for k, (lo, hi) in KEYWORD.items()]
    pd.DataFrame(lm).to_csv(out / "level_map.csv", index=False, encoding="utf-8-sig")
    print("\n".join(rep))
    print(f"\n-> {out/'jobs_clean.csv'} ({len(df)} dòng)")


if __name__ == "__main__":
    main()