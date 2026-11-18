"""
build_skill_taxonomy.py  -  taxonomy kỹ năng, bảng ánh xạ biến thể và ma trận binary skill

Chạy: python pipeline/build_skill_taxonomy.py --input data_private/jobs_clean.csv

THUẬT TOÁN
  1. Chuẩn hóa token trong skills_raw (bỏ dấu, chữ thường, bỏ ký tự không phải chữ/số/+/#) rồi ánh xạ sang canonical
     bằng TAXONOMY (kèm alias). Token nhiễu (NOISE) bị bỏ; MULTI cho một token ứng với nhiều canonical.
  2. Roll-up cha-con (PARENTS, suy ra từ MULTI): có skill con thì tính là có skill cha.
  3. Ngoại ngữ / soft skill lấy từ văn bản JD (title + requirements + description), không lấy từ skills_raw.
     Đây là nhánh ĐỘC LẬP với bước 1-2 (chỉ gặp nhau ở bước hợp tập). Danh sách ngoại ngữ do TAXONOMY định nghĩa
     (LANGUAGES); regex nhận diện ở LANG_LATIN (văn bản đã bỏ dấu)
  4. Với mỗi hard skill tính số job, % trên tổng và theo từng nhóm nghề:
         prevalence = % job của nhóm có skill
         lift       = prevalence trong nhóm / prevalence ở các nhóm còn lại
  5. Skill vào ma trận binary nếu thỏa mãn MỘT trong các quy tắc:
         (a) global        : số job >= --min-jobs (mặc định 50)
         (b) role_core     : ở một nhóm nghề có prevalence >= --role-min-pct (8%), >= --role-min-jobs (10) job
                             và lift >= --role-min-lift (1,5)
         (c) language      : ngoại ngữ trong taxonomy có số job >= --lang-min-jobs (15) -> cột riêng; ngoại ngữ ít hơn
                             ngưỡng này không vào binary (vẫn nằm trong taxonomy và được tính vào n_languages)
         (d) prog_language : ngôn ngữ lập trình có >= --prog-min-jobs (25) job
  6. Gộp CISSP / CEH / OSCP / CISM thành cột skill_cert_security, thêm n_languages, ghép metadata và ghi jd_skill_binary.csv
     cùng taxonomy, alias map, unmapped, implied pairs, report.
"""
# Khai báo thư viện
import argparse
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# DEFAULT_ROLES = ["Backend", "Frontend", "Data Engineer", "Data Scientist", "AI Engineer", "DevOps/SRE", "Mobile", "QA/Tester", "BA/PM", "Security", "Cloud"]
CERT_GROUP = {"skill_cert_security": ["CISSP", "CEH", "OSCP", "CISM"]}
META_COLS = ["job_uid", "source", "job_title", "company_name", "city", "expertise_category",
             "level_group", "exp_min", "exp_max", "exp_group",
             "job_url", "crawled_at"]
OUT_NAME = "jd_skill_binary.csv"

# Chuẩn hóa chuỗi (bỏ dấu, chữ thường, bỏ ký tự không phải chữ/số/+/#)
def key(s: str) -> str:
    s = str(s).replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9+#]", "", s)

# Khai báo taxonomy, alias, meta, các skill nhóm mềm, skill ngoại ngữ, skill chứng chỉ, skill lập trình
# Cấu trúc TAXONOMY = {("Group", "Subgroup"): {"Canonical": ["alias1", "alias2"]}}
TAXONOMY = {
    ("Programming Language", "Language"): {
        "Python": [], "Java": [], "JavaScript": ["js", "es6", "ecmascript"], "TypeScript": ["ts"],
        "Go": ["golang"], "C#": ["csharp"], "C++": ["cpp"], "C": ["c language", "c programming"],
        "PHP": [], "Ruby": [], "Kotlin": [], "Swift": [], "Objective-C": ["objc"], "Scala": [], "Rust": [],
        "Dart": [], "R": ["r language", "r programming"], "Bash/Shell": ["bash", "shell", "shell scripting", "bash shell", "sh"],
        "PowerShell": [], "VBA": [],
    },
    ("Frontend", "Framework/Markup"): {
        "React": ["reactjs"], "Angular": ["angularjs"], "Vue": ["vuejs", "vue 3"], "Next.js": ["next"],
        "Nuxt.js": ["nuxt"], "HTML": ["html5"], "CSS": ["css3"], "Sass/SCSS": ["sass", "scss"],
        "Tailwind CSS": ["tailwind"], "Bootstrap": [], "Redux": ["redux toolkit"], "Webpack": [], "Vite": [], "jQuery": [],
    },
    ("Backend Framework", "Framework"): {
        "Spring": ["spring framework", "spring mvc", "spring cloud", "spring data", "spring security"],
        "Spring Boot": ["springboot"], "Hibernate/JPA": ["hibernate", "jpa"], "MyBatis": [],
        ".NET": [".net core", "dotnet", ".net framework", "asp.net", "asp.net core", "asp.net mvc"],
        "Node.js": ["node"], "NestJS": [], "Express": ["express.js"], "Django": [], "Flask": [], "FastAPI": [],
        "Laravel": [], "Ruby on Rails": ["rails", "ror"],
    },
    ("Mobile", "Platform/Framework"): {
        "Android": [], "iOS": [], "Flutter": [], "React Native": [], "Jetpack Compose": [], "SwiftUI": [], "Xcode": [],
    },
    ("Database & Storage", "Database"): {
        "SQL": ["t-sql", "sql query"], "MySQL": [], "PostgreSQL": ["postgres", "postgre"],
        "Oracle": ["oracle db", "oracle database", "pl/sql"],
        "SQL Server": ["ms sql", "mssql", "microsoft sql server", "microsoft azure sql database", "ms sql server"],
        "MongoDB": ["mongo"], "Redis": [], "NoSQL": [], "Elasticsearch": ["elastic search", "elastic", "opensearch"],
        "Cassandra": [], "DynamoDB": [], "SQLite": [], "MariaDB": [], "Neo4j": [], "ClickHouse": [],
    },
    ("Data Engineering", "Pipeline/Platform"): {
        "ETL/ELT": ["etl", "elt", "data pipeline", "data pipelines"], "Airflow": ["apache airflow", "cloud composer"],
        "Spark": ["apache spark", "pyspark"], "Kafka": ["apache kafka"], "Flink": ["apache flink"], "dbt": [],
        "Data Warehouse": ["data warehousing", "dwh"], "Data Lake": ["lakehouse", "data lakehouse"],
        "Data Modeling": ["data modelling"], "Big Data": [], "NiFi": ["apache nifi"], "Talend": [], "Informatica": [],
        "SSIS": [], "Snowflake": [], "BigQuery": ["google bigquery"], "Redshift": [], "Databricks": [], "Hadoop": ["apache hadoop"],
        "Hive": [], "Azure Synapse": ["synapse"], "Microsoft Fabric": ["fabric"], "Data Quality": ["data quality tools", "great expectations"],
        "RabbitMQ": [], "Message Queue": ["mq", "message broker", "messaging"], "Pub/Sub": ["pubsub", "google pub/sub"],
    },
    ("Data Analytics & BI", "Analytics/BI"): {
        "Power BI": ["powerbi", "microsoft power bi", "power query", "dax"], "Tableau": ["tableu"],
        "Looker": ["looker studio", "google data studio", "data studio"], "Metabase": [], "Superset": ["apache superset"],
        "Qlik": ["qlikview", "qlik sense"], "Excel": ["microsoft excel", "google sheets", "spreadsheet", "advanced excel"],
        "Data Analysis": ["data analytics", "analytics"], "Business Intelligence": ["bi", "bi tools"],
        "A/B Testing": ["ab testing", "a/b test", "experimentation"], "Statistics": ["statistical analysis"],
        "Data Visualization": ["data viz", "visualization", "dashboard", "dashboards"], "Data Mining": [],
    },
    ("AI/ML", "ML/AI"): {
        "Machine Learning": ["ml"], "Deep Learning": ["dl"], "NLP": ["natural language processing"],
        "Computer Vision": ["cv","image processing", "opencv"], "LLM": ["llms", "large language model", "large language models"],
        "Generative AI": ["genai", "gen ai"], "RAG": ["retrieval augmented generation"],
        "Agentic AI": ["ai agent", "ai agents", "agentic", "multi-agent"], "Prompt Engineering": [],
        "LangChain": ["langgraph"], "Hugging Face": ["transformers"], "PyTorch": [], "TensorFlow": ["keras"],
        "scikit-learn": ["sklearn"], "Pandas": [], "NumPy": [], "MLOps": [], "MLflow": [], "Kubeflow": [],
        "Vertex AI": [], "SageMaker": ["amazon sagemaker"], "OpenAI": ["openai api", "gpt"], "OCR": [],
        "Data Science": ["data scientist"], "Artificial Intelligence": ["ai", "artificial intelligence", "ai/ml"],
    },
    ("AI Dev Tools", "AI coding assistant"): {
        "Claude": ["claude code", "anthropic"], "GitHub Copilot": ["copilot", "microsoft copilot"], "Cursor": ["cursor ai"],
        "ChatGPT": [], "Gemini": ["google gemini"], "Windsurf": [],
    },
    ("Cloud", "Platform"): {
        "AWS": ["amazon web services", "s3", "ec2", "lambda", "aws lambda", "cloudformation", "aws cloudformation"],
        "Azure": ["microsoft azure"], "GCP": ["google cloud", "google cloud platform"], "Alibaba Cloud": ["aliyun"],
        "Firebase": [], "Supabase": [], "Serverless": [],
        "Cloud Computing": ["cloud", "cloud native", "cloud-native architecture", "cloud infrastructure", "hybrid cloud", "public cloud", "cloud migration"],
    },
    ("DevOps & Infra", "Tooling"): {
        "CI/CD": ["continuous integration", "continuous delivery", "continuous deployment"],
        "Docker": ["containerization", "containers"], "Kubernetes": ["k8s", "aks", "gke", "openshift"],
        "Terraform": [], "Infrastructure as Code": ["iac"], "Ansible": [], "Jenkins": [],
        "GitHub Actions": ["github action"], "GitLab CI": ["gitlab ci/cd", "gitlab pipelines"], "ArgoCD": ["argo cd", "argo"],
        "Helm": [], "Prometheus": [], "Grafana": [], "Datadog": [], "ELK": ["elk stack", "elastic stack", "kibana", "logstash"],
        "Splunk": [], "OpenTelemetry": ["otel"], "Observability": ["monitoring", "logging", "apm"], "CircleCI": [],
        "Linux": ["unix", "linux administration"], "Nginx": [], "Istio": ["service mesh"], "Vault": ["hashicorp vault"],
        "Pulumi": [], "VMware": ["vsphere"], "Networking": ["network", "tcp/ip"], "DevOps": ["sre", "site reliability"],
    },
    ("Security", "Security"): {
        "Cybersecurity": ["cyber security", "information security", "infosec", "security", "it security", "network security"],
        "SIEM": ["wazuh", "qradar"], "SOC": ["security operations center", "soc analyst"],
        "Penetration Testing": ["pentest", "pen test", "pentesting", "vapt", "red team", "ethical hacking"],
        "OWASP": ["owasp top 10"], "Burp Suite": ["burp"], "Metasploit": [], "Nmap": [], "Wireshark": [], "WAF": [], "DLP": [],
        "IDS/IPS": ["ids", "ips", "intrusion detection"], "EDR": ["xdr"], "SOAR": [], "Firewall": ["firewalls"],
        "Zero Trust": ["zero trust architecture"], "IAM": ["identity & access management", "identity and access management", "pam", "okta", "keycloak"],
        "OAuth": ["oauth2", "openid connect", "oidc"], "JWT": [], "SSO": [], "SAML": [],
        "Cloud Security": [], "Application Security": ["appsec"], "SAST": [], "DAST": [], "DevSecOps": [],
        "Incident Response": ["dfir"], "Vulnerability Management": ["vulnerability assessment", "vulnerability scanning"],
        "ISO 27001": ["iso/iec 27001"], "PCI DSS": [], "SOC 2": [], "NIST": ["nist csf"],
        "GRC": ["governance, risk & compliance", "governance", "risk & compliance", "compliance"],
        "Risk Management": [], "Data Privacy": ["data privacy / compliance", "gdpr", "privacy"],
    },
    ("Certification", "Cert"): {
        "CISSP": [], "ITIL": ["itil foundation"], "CEH": [], "OSCP": [], "CISM": [], "CISA": [], "CompTIA": ["security+", "comptia security+"], "TOGAF": [],
    },
    ("Testing/QA", "Testing"): {
        "Software Testing": ["qa qc", "qa", "qc", "tester", "testing", "manual testing", "manual test", "software tester", "quality assurance"],
        "Automation Testing": ["automation test", "test automation", "automated testing"],
        "Selenium": [], "Cypress": [], "Playwright": [], "Appium": [], "JMeter": ["apache jmeter"], "Postman": [],
        "JUnit": [], "Mockito": [], "Jest": [], "pytest": [], "Robot Framework": [], "k6": [],
        "Unit Testing": ["unit test", "unit tests"], "Integration Testing": ["integration test"],
        "Performance Testing": ["load testing", "performance test"],
    },
    ("Architecture & API", "Design"): {
        "Microservices": ["microservice", "microservices architecture"], "Event-Driven": ["event driven", "event-driven architecture", "eda"],
        "REST API": ["api", "apis", "restful", "restful api", "rest", "web api", "api development", "api design"],
        "GraphQL": [], "gRPC": [], "WebSocket": ["websockets"], "SOAP": [], "OpenAPI/Swagger": ["openapi", "swagger"],
        "Software Architecture": ["system architecture", "solution architecture", "technical architecture", "system design",
                                  "architecture", "solution design", "software architect"],
        "Enterprise Architecture": [], "Domain-Driven Design": ["ddd"], "CQRS": [], "Clean Architecture": [],
        "SOLID": ["solid principles"], "OOP": ["object-oriented programming", "object oriented"], "Design Patterns": ["design pattern"],
        "MVC": [], "MVVM": [], "Monolith": [], "UML": [], "Middleware": [],
        "Data Structures & Algorithms": ["algorithms", "data structures", "dsa"],
    },
    ("Methodology & Tools", "Process/Tool"): {
        "Agile": ["agile methodology"], "Scrum": [], "Kanban": [], "Waterfall": ["waterfall methodology"], "SDLC": [],
        "Jira": [], "Confluence": [], "Trello": [], "Slack": [], "Git": ["version control"], "GitHub": [], "GitLab": [],
        "Bitbucket": [], "Maven/Gradle": ["maven", "gradle"], "PowerPoint": ["microsoft powerpoint"],
    },
    ("BA/PM & Product", "BA/PM"): {
        "Business Analysis": ["business analyst", "ba", "requirement analysis", "requirements analysis", "requirement gathering",
                              "requirements gathering", "requirements elicitation"],
        "BRD": [], "SRS": [], "User Story": ["user stories"], "Use Case": ["use cases"], "BPMN": ["business process modeling"],
        "Wireframing": ["wireframe", "prototype", "prototyping", "mockup"], "UAT": [],
        "Project Management": ["project manager", "pmp"], "Product Management": ["product manager", "product strategy"],
        "Product Owner": ["po"], "Presales": ["presale"], "Consulting": ["it consulting"],
        "Diagramming Tools": ["visio", "draw.io"],
    },
    ("Business & Domain", "Platform"): {
        "ERP": [], "SAP": ["abap"], "CRM": [], "Salesforce": [], "Odoo": [], "Shopify": [], "WordPress": [],
        "HRM/HCM": ["hrm", "hcm", "hris"], "Core Banking": [], "Power Automate": ["power apps"],
    },
    ("Other Tech", "Other"): {
        "Blockchain": ["web3", "smart contract", "smart contracts", "solidity"], "Embedded": ["embedded systems", "firmware"],
        "Unity": [], "IT Support": ["helpdesk"], "System Administration": ["system admin", "sysadmin", "windows server", "active directory"],
    },
    ("Design/UI-UX", "Design"): {
        "Figma": [], "UI/UX": ["ux", "ui", "ux design", "ui design", "user experience", "product design"],
        "Design Systems": ["design system"], "Adobe Creative Suite": ["adobe", "photoshop", "adobe photoshop", "illustrator"],
    },
    ("Soft Skill", "Language"): {
        "English": ["tieng anh", "ielts", "toeic", "toefl", "english communication", "fluent english"],
        "Japanese": ["japanese it communication", "jlpt", "tieng nhat"], "Chinese": ["mandarin", "tieng trung", "hsk"],
        "French": ["tieng phap"], "Korean": ["tieng han", "topik"], "German": ["tieng duc"],
        "Spanish": ["tieng tay ban nha"], "Russian": ["tieng nga"], "Italian": ["tieng italia"],
        "Portuguese": ["tieng bo dao nha"], "Thai": ["tieng thai"], "Indonesian": ["bahasa indonesia", "tieng indonesia"],
        "Arabic": ["tieng a rap"],
    },
    ("Soft Skill", "Interpersonal"): {
        "Communication": ["communication skills", "giao tiep", "interpersonal skills", "presentation skills"],
        "Teamwork": ["team work", "team player", "collaboration", "lam viec nhom"], "Adaptability": ["adaptable", "flexibility"],
    },
    ("Soft Skill", "Cognitive"): {
        "Problem Solving": ["giai quyet van de"], "Critical Thinking": ["analytical thinking", "analytical skills", "logical thinking"],
        "Time Management": ["organization", "organizational skills"], "Attention to Detail": ["detail-oriented"],
    },
    ("Soft Skill", "Management"): {
        "Leadership": ["leadership skills", "lanh dao"], "Team Management": ["people management", "team leadership", "mentoring", "coaching"],
        "Stakeholder Management": ["stakeholder", "stakeholders"],
    },
}
# Danh sách ngoại ngữ lấy trực tiếp từ taxonomy (nguồn sự thật duy nhất): thêm ngôn ngữ = sửa TAXONOMY + LANG_LATIN/LANG_NATIVE
LANGUAGES = list(TAXONOMY[("Soft Skill", "Language")])
# Khai báo quy tắc bổ trợ (một token có thể tương ứng với nhiều canonical trong taxonomy)
MULTI = {
    "eks": ["Kubernetes", "AWS"], "ecs": ["AWS", "Docker"], "embedded c": ["C", "Embedded"],
    "gitlab ci": ["GitLab CI", "CI/CD"], "github actions": ["GitHub Actions", "CI/CD"], "jenkins": ["Jenkins", "CI/CD"],
    "next.js": ["Next.js", "React"], "spring boot": ["Spring Boot", "Spring"],
    "pyspark": ["Spark", "Python"], "sagemaker": ["SageMaker", "AWS"], "redshift": ["Redshift", "AWS", "Data Warehouse"],
    "dynamodb": ["DynamoDB", "AWS", "NoSQL"], "bigquery": ["BigQuery", "GCP", "Data Warehouse"],
    "snowflake": ["Snowflake", "Data Warehouse"], "azure synapse": ["Azure Synapse", "Azure", "Data Warehouse"],
    "vertex ai": ["Vertex AI", "GCP"],
}
PARENTS = {c[0]: c[1:] for t, c in MULTI.items() if len(c) > 1 and key(t) == key(c[0])} # Xác định parent cho các canonical có cùng alias
# Khai báo các token nhiễu, các token cần bỏ, các tiền tố nhiễu và regex để loại bỏ các token nhiễu (trừ Governance)
NOISE = {key(x) for x in [
    "Engineering and Information Technology", "Information Technology", "Information Technology and Engineering", "Engineering",
    "Information Technology and Consulting", "Art/Creative", "Management and Manufacturing", "Manufacturing", "Sales", "Marketing",
    "Public Relations", "Other", "Games", "Design", "Database", "Fullstack", "Data Engineer", "Mobile Apps", "Writing/Editing",
    "Governance", "Finance", "Healthcare",
]} - {key("Governance")}
DROP_TAGS = {key("Fresher Accepted"), key("Internship Accepted")}
NOISE_PREFIX = ("and ",)
NOISE_RE = re.compile(r"information technology|general business|business development|education and training|finance and sales|^apache$|^data-driven$", re.I)

# Nhận diện NGOẠI NGỮ từ văn bản JD. Hai tập mẫu, khóa phải trùng tên canonical trong TAXONOMY (có assert ở main):
#   LANG_LATIN  : khớp trên văn bản đã bỏ dấu (tiếng Anh + tiếng Việt không dấu)
#   LANG_NATIVE : khớp trên văn bản GỐC, vì chữ Hán / Hangul / Kana bị mất khi bỏ dấu bằng ASCII
# Lưu ý: "Thái" sau khi bỏ dấu trùng tên riêng tiếng Việt (Thái Nguyên, Thái Bình...) nên "Thai" chỉ khớp khi đi kèm "tiếng"/"language".
LANG_LATIN = {
    "English": r"\b(english|tieng anh|ielts|toeic|toefl)\b",
    "Japanese": r"\b(japanese|tieng nhat|jlpt)\b",
    "Korean": r"\b(korean|tieng han|topik)\b",
    "Chinese": r"\b(chinese|mandarin|tieng trung|hsk)\b",
    "French": r"\b(french|tieng phap)\b",
    "German": r"\b(german|tieng duc)\b",
    "Spanish": r"\b(spanish|tieng tay ban nha)\b",
    "Russian": r"\b(russian|tieng nga)\b",
    "Italian": r"\b(italian|tieng italia)\b",
    "Portuguese": r"\b(portuguese|tieng bo dao nha)\b",
    "Thai": r"\b(tieng thai|thai language)\b",
    "Indonesian": r"\b(indonesian|bahasa indonesia|tieng indonesia)\b",
    "Arabic": r"\b(arabic|tieng a rap)\b",
}
LANG_NATIVE = {
    "Japanese": r"日本語|日语|日語",
    "Korean": r"한국어|韩语|韓語",
    "Chinese": r"中文|汉语|漢語|普通话|普通話",
}
# Soft skill KHÔNG phải ngoại ngữ, lấy từ văn bản JD (title + requirements + description) bằng regex
SOFT_TEXT = {
    "Communication": r"\b(communication skills?|giao tiep|interpersonal|presentation skills?)\b",
    "Teamwork": r"\b(team ?work|team player|lam viec nhom|lam viec theo nhom)\b",
    "Adaptability": r"\b(adaptab\w+|fast learner|quick learner|willingness to learn|kha nang thich nghi|ham hoc hoi)\b",
    "Problem Solving": r"\b(problem[- ]?solving|giai quyet van de)\b",
    "Critical Thinking": r"\b(critical thinking|analytical (thinking|skills?)|logical thinking|tu duy (logic|phan tich|phan bien))\b",
    "Time Management": r"\b(time management|quan ly thoi gian|organi[sz]ational skills?)\b",
    "Attention to Detail": r"\b(attention to detail|detail[- ]oriented|can than)\b",
    "Leadership": r"\b(leadership|lanh dao)\b",
    "Team Management": r"\b((team|people) management|manage[sd]? (a |the |our )?(engineering |technical )?teams?|mentor(ing|s)? (junior|team|engineers|others)|coach(ing)? (junior|team)|quan ly nhom)\b",
    "Stakeholder Management": r"\b(stakeholder (management|engagement|communication)|manage stakeholders|quan ly stakeholder)\b",
} # Biểu thức mẫu regex dạng chuỗi
SOFT_RX = {c: re.compile(p, re.I) for c, p in SOFT_TEXT.items()} # Biểu thức mẫu regex đã biên dịch từ chuỗi
LANG_RX = {c: re.compile(p, re.I) for c, p in LANG_LATIN.items()}
LANG_NATIVE_RX = {c: re.compile(p) for c, p in LANG_NATIVE.items()}

# Bỏ dấu tiếng Việt, về ASCII, gộp khoảng trắng (dùng cho LANG_LATIN và SOFT_TEXT)
def _fold(text):
    t = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t)

# Trích soft skill + ngoại ngữ từ văn bản
def soft_from_text(text):
    t = _fold(text)
    raw = re.sub(r"\s+", " ", text)
    found = {c for c, rx in SOFT_RX.items() if rx.search(t)}
    found |= {c for c, rx in LANG_RX.items() if rx.search(t)}
    found |= {c for c, rx in LANG_NATIVE_RX.items() if rx.search(raw)}
    return found

# Đoạn trích quanh từ khóa khớp (phục vụ kiểm tra tay precision, xem --audit-lang)
def lang_matches(text, lang):
    out = []
    for rx, src in ((LANG_RX.get(lang), _fold(text)), (LANG_NATIVE_RX.get(lang), text)):
        if rx is not None:
            out += [src[max(0, m.start() - 40): m.end() + 40].replace("\n", " ") for m in rx.finditer(src)]
    return out

# Xây dựng alias cho các skill (ví dụ "js" -> "JavaScript", "reactjs" -> "React", "python" -> "Python")
def build_alias():
    alias, meta = {}, {}
    for (group, sub), skills in TAXONOMY.items():
        for canon, alts in skills.items():
            meta[canon] = {"group": group, "subgroup": sub, "type": "soft" if group == "Soft Skill" else "hard", "aliases": alts}
            for a in [canon] + alts:
                k = key(a)
                if k in alias and alias[k] != [canon]:
                    raise ValueError(f"alias '{a}' trùng giữa {alias[k]} và {canon}")
                alias[k] = [canon]
    for a, canons in MULTI.items():
        assert all(c in meta for c in canons), f"MULTI có canonical lạ: {canons}"
        alias[key(a)] = canons
    return alias, meta

# Chuẩn hóa tên cột canonical (dùng cho ma trận)
def slug(canon, prefix):
    s = canon.lower().replace("c++", "cpp").replace("c#", "csharp").replace(".net", "dotnet")
    return prefix + re.sub(r"[^a-z0-9]+", "_", s).strip("_")

# Nhận tham số và kiểm tra dữ liệu đầu vào và xuất kết quả
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_private/jobs_clean.csv")
    ap.add_argument("--outdir", default="data_private")
    ap.add_argument("--min-jobs", type=int, default=50, help="quy tắc (a) global: số job tối thiểu trên toàn dữ liệu để 1 skill vào binary (~2%%)")
    ap.add_argument("--roles", default="ALL", help="'ALL' (mặc định, dùng MỌI nhóm nghề có trong dữ liệu cho quy tắc role_core) hoặc danh sách cách nhau bằng dấu phẩy")
    ap.add_argument("--role-min-pct", type=float, default=0.08, help="quy tắc (b) role_core: %% tối thiểu trong 1 nhóm nghề (0.08 = 8%%)")
    ap.add_argument("--role-min-jobs", type=int, default=10, help="quy tắc (b) role_core: số job tối thiểu có skill đó trong nhóm nghề")
    ap.add_argument("--role-min-lift", type=float, default=1.5, help="quy tắc (b) role_core: tỷ lệ trong nhóm phải cao hơn ít nhất bấy nhiêu lần so với các nhóm còn lại")
    ap.add_argument("--prog-min-jobs", type=int, default=25, help="quy tắc (d): ngôn ngữ lập trình có >= số job này thì vào binary")
    ap.add_argument("--lang-min-jobs", type=int, default=25, help="quy tắc (c): ngoại ngữ có >= số job này được giữ cột riêng; ít hơn thì không vào binary")
    ap.add_argument("--audit-lang", type=int, default=0, help="N > 0: ghi skill_lang_audit.csv (mẫu N job mỗi ngoại ngữ, kèm đoạn trích) để kiểm tay precision; 0 = tắt")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    need = list(dict.fromkeys(META_COLS + ["skills_raw", "requirements_raw", "description_raw"]))
    head = pd.read_csv(a.input, nrows=0).columns
    # Kiểm tra dữ liệu đầu vào có đủ cột cần thiết
    miss = [c for c in need if c not in head]
    if miss:
        raise SystemExit(f"{a.input} thiếu cột: {miss} -- kiểm tra đã chạy clean_jobs.py bản mới nhất chưa.")
    df = pd.read_csv(a.input, usecols=need)
    if df.job_uid.duplicated().any(): # kiểm tra trùng job_uid
        raise SystemExit(f"job_uid trùng lặp: {int(df.job_uid.duplicated().sum())} dòng -- sửa jobs_clean trước khi chạy.")
    roles = (sorted(df.expertise_category.dropna().unique(), key=lambda r: -(df.expertise_category == r).sum())
             if a.roles == "ALL" else [r.strip() for r in a.roles.split(",") if r.strip()])
    alias, meta = build_alias()
    assert all(c in meta for c in SOFT_TEXT), "SOFT_TEXT có canonical lạ"
    assert set(LANG_LATIN) == set(LANGUAGES), f"LANG_LATIN lệch taxonomy: {sorted(set(LANG_LATIN) ^ set(LANGUAGES))}"
    assert set(LANG_NATIVE) <= set(LANGUAGES), f"LANG_NATIVE có ngôn ngữ lạ: {sorted(set(LANG_NATIVE) - set(LANGUAGES))}"
    # ---Ánh xạ token cho từng JD---
    var_cnt = Counter()
    unmapped = Counter()
    job_skills = {}
    noise_cnt, drop_cnt = Counter(), Counter()
    for uid, val, ti, rq, ds in zip(df.job_uid, df.skills_raw, df.job_title, df.requirements_raw, df.description_raw):
        sk = set()
        if isinstance(val, str):
            for tok in re.split(r"\s*\|\s*", val):
                tok = tok.strip()
                if not tok:
                    continue
                k = key(tok)
                if k in alias:
                    sk.update(alias[k]); var_cnt[(tok, "|".join(alias[k]))] += 1
                elif k in DROP_TAGS:
                    drop_cnt[tok] += 1
                elif k in NOISE or tok.lower().startswith(NOISE_PREFIX) or NOISE_RE.search(tok):
                    noise_cnt[tok] += 1
                else:
                    unmapped[tok] += 1
        for c in list(sk):
            sk.update(PARENTS.get(c, []))
        sk = {c for c in sk if meta[c]["type"] == "hard"}
        # Nhánh văn bản JD độc lập với skills_raw nhưng phải cộng SAU bộ lọc hard ở trên, nếu không soft skill bị loại
        sk |= soft_from_text(" \n ".join(x for x in (ti, rq, ds) if isinstance(x, str)))
        job_skills[uid] = sk

    has = df.skills_raw.notna() & (df.skills_raw.str.strip() != "")
    n_jobs = int(has.sum())
    src = dict(zip(df.job_uid, df.source))

    # ---- thống kê theo canonical ----
    cnt, by_src, by_role = Counter(), defaultdict(Counter), defaultdict(Counter)
    role_of = dict(zip(df.job_uid, df.expertise_category))
    has_map = dict(zip(df.job_uid, has))
    for uid, sk in job_skills.items():
        for c in sk:
            cnt[c] += 1; by_src[c][src[uid]] += 1
            r = role_of[uid]
            if meta[c]["type"] == "hard" and pd.notna(r):
                by_role[c][r] += 1
    role_tot_all = df[has & df.expertise_category.notna()].expertise_category.value_counts().to_dict()
    all_roles_present = list(role_tot_all.keys())
    n_has_total = sum(role_tot_all.values())
    src_tot_h = df[has].source.value_counts().to_dict()
    src_tot_a = df.source.value_counts().to_dict()
    rows = []
    for c, m in meta.items():
        n = cnt.get(c, 0)
        den, st = (n_jobs, src_tot_h) if m["type"] == "hard" else (len(df), src_tot_a)
        shares = {s: by_src[c].get(s, 0) / st[s] for s in st}
        # --- quy tắc (b) role_core: xét TỪNG nhóm nghề trong all_roles_present, so với "các nhóm còn lại" ---
        qualifying = []
        if m["type"] == "hard":
            for r in all_roles_present:
                n_r = by_role[c].get(r, 0)
                tot_r = role_tot_all[r]
                p_r = n_r / tot_r if tot_r else 0
                n_other = sum(by_role[c].get(r2, 0) for r2 in all_roles_present if r2 != r)
                tot_other = n_has_total - tot_r
                p_other = n_other / tot_other if tot_other else 0
                lift = (p_r / p_other) if p_other > 0 else (np.inf if p_r > 0 else 0)
                if n_r >= a.role_min_jobs and p_r >= a.role_min_pct and lift >= a.role_min_lift:
                    qualifying.append((r, p_r, lift))
        qualifying.sort(key=lambda t: -t[2])
        is_global = bool(m["type"] == "hard" and n >= a.min_jobs)
        is_role_core = bool(qualifying)
        is_lang = c in LANGUAGES and n >= a.lang_min_jobs
        is_prog = bool(m["group"] == "Programming Language" and n >= a.prog_min_jobs and not (is_global or is_role_core))
        reason = "+".join(x for x in ("global" if is_global else "", "role_core" if is_role_core else "",
                                      "language" if is_lang else "", "prog_language" if is_prog else "") if x)
        best_role, best_pct, best_lift = qualifying[0] if qualifying else (None, None, None)
        rows.append({"canonical": c, "group": m["group"], "subgroup": m["subgroup"], "type": m["type"],
                     "signal": "skills_raw" if m["type"] == "hard" else "jd_text",
                     "n_jobs": n, "df_pct": round(100 * n / den, 2),
                     **{f"df_{s}_pct": round(100 * v, 1) for s, v in shares.items()},
                     "tier": "A (>=5%)" if n / den >= .05 else "B (2-5%)" if n / den >= .02 else "C (1-2%)" if n / den >= .01 else "D (<1%)",
                     "is_role_defining": is_role_core,
                     "role_core_best_role": best_role, "role_core_best_pct": round(100 * best_pct, 1) if best_pct is not None else None,
                     "role_core_best_lift": round(best_lift, 2) if best_lift not in (None, np.inf) else (best_lift if best_lift == np.inf else None),
                     "role_core_all": "; ".join(f"{r} ({100*p:.1f}%, lift {l:.2f})" if l != np.inf else f"{r} ({100*p:.1f}%, lift inf)" for r, p, l in qualifying),
                     "in_binary": bool(reason), "binary_reason": reason, "aliases": "; ".join(m["aliases"])})
    tax = pd.DataFrame(rows).sort_values(["type", "n_jobs"], ascending=[True, False])
    tax.to_csv(out / "skill_taxonomy.csv", index=False, encoding="utf-8-sig")
    

    # ---- alias map ----
    am = pd.DataFrame([{"raw_variant": r, "canonical": c, "n_occurrences": n,
                        "group": meta[c.split("|")[0]]["group"], "type": meta[c.split("|")[0]]["type"]}
                       for (r, c), n in var_cnt.items()]).sort_values(["canonical", "n_occurrences"], ascending=[True, False])
    am.to_csv(out / "skill_alias_map.csv", index=False, encoding="utf-8-sig")
    um = pd.DataFrame(unmapped.most_common(), columns=["token", "n"])
    um.to_csv(out / "skill_unmapped.csv", index=False, encoding="utf-8-sig")

    # ---- binary ----
    sel = tax[tax.in_binary]
    cert_members = [c for c in CERT_GROUP["skill_cert_security"] if c in meta]
    hard_cols = {}
    for _, r in sel[sel.type == "hard"].iterrows():
        if r.canonical in cert_members:
            continue
        col = slug(r.canonical, "skill_")
        assert col not in hard_cols, f"trùng tên cột {col}: {r.canonical} / {hard_cols[col]}"
        hard_cols[col] = r.canonical
    lang_keep = set(sel.loc[sel.binary_reason.str.split("+").apply(lambda xs: "language" in xs), "canonical"])
    lang_cols = {slug(l, "skill_"): l for l in LANGUAGES if l in lang_keep}
    for col in lang_cols:
        assert col not in hard_cols, f"trùng tên cột {col}"
    bin_rows = []
    for uid in df.job_uid:
        sk = job_skills[uid]
        hs = {c for c in sk if meta[c]["type"] == "hard"}
        ok = has_map[uid]
        row = {"job_uid": uid, "has_skills": int(ok),
               "n_hard_skills": len(hs) if ok else None,
               "n_skill_groups": len({meta[c]["group"] for c in hs}) if ok else None,
               "skills_status": "none" if not ok else ("has_hard" if hs else "noise_only")}
        row["n_languages"] = sum(l in sk for l in LANGUAGES)
        row.update({col: int(l in sk) for col, l in lang_cols.items()})
        row.update({col: int(c in sk) for col, c in hard_cols.items()})
        row["skill_cert_security"] = int(any(c in sk for c in cert_members))
        bin_rows.append(row)
    bin_df = pd.DataFrame(bin_rows)
    assert (bin_df.job_uid.values == df.job_uid.values).all()
    final = pd.concat([df[META_COLS].reset_index(drop=True), bin_df.drop(columns="job_uid")], axis=1)
    final.to_csv(out / OUT_NAME, index=False, encoding="utf-8-sig")

    # ---- (tùy chọn) skill_lang_audit.csv: mẫu job theo ngoại ngữ để kiểm tay precision regex ----
    if a.audit_lang > 0:
        txt = {u: " \n ".join(x for x in (ti, rq, ds) if isinstance(x, str))
               for u, ti, rq, ds in zip(df.job_uid, df.job_title, df.requirements_raw, df.description_raw)}
        rng = np.random.default_rng(42)
        audit = []
        for l in LANGUAGES:
            ids = [u for u, sk in job_skills.items() if l in sk]
            pick = rng.choice(ids, size=min(a.audit_lang, len(ids)), replace=False) if ids else []
            for u in pick:
                audit.append({"language": l, "n_jobs_language": len(ids), "job_uid": u,
                              "snippet": " ... ".join(lang_matches(txt[u], l)[:2]), "is_correct": ""})
        pd.DataFrame(audit, columns=["language", "n_jobs_language", "job_uid", "snippet", "is_correct"]).to_csv(
            out / "skill_lang_audit.csv", index=False, encoding="utf-8-sig")
        print(f"Ghi skill_lang_audit.csv: {len(audit)} dòng (điền cột is_correct để tính precision từng ngoại ngữ).")

    # ---- skill_implied_pairs.csv: cặp con -> cha do luật roll-up (PARENTS) tự tạo ra, 100% theo cấu trúc ----
    final_cols = set(final.columns)
    ip_rows = []
    for child, parents in sorted(PARENTS.items()):
        n_child = sum(child in sk for sk in job_skills.values())
        for p in parents:
            n_both = sum(child in sk and p in sk for sk in job_skills.values())
            ip_rows.append({"child": slug(child, "skill_"), "implied_parent": slug(p, "skill_"),
                            "child_canonical": child, "parent_canonical": p, "n_child_jobs": n_child,
                            "pct_parent_given_child": round(100 * n_both / n_child, 1) if n_child else None,
                            "child_in_binary": slug(child, "skill_") in final_cols,
                            "parent_in_binary": slug(p, "skill_") in final_cols})
    pd.DataFrame(ip_rows, columns=["child", "implied_parent", "child_canonical", "parent_canonical", "n_child_jobs",
                                   "pct_parent_given_child", "child_in_binary", "parent_in_binary"]).to_csv(
        out / "skill_implied_pairs.csv", index=False, encoding="utf-8-sig")
    print(f"Ghi skill_implied_pairs.csv: {len(ip_rows)} cặp con -> cha theo luật roll-up.")
    n_skill_cols = sum(c.startswith("skill_") for c in final.columns)
    print(f"Ghi {OUT_NAME}: {len(final)} job, {n_skill_cols} cột skill_*, {final.shape[1] - n_skill_cols} cột metadata/thống kê.")

    # ---- report ----
    hard = tax[tax.type == "hard"]; soft = tax[tax.type == "soft"]
    tot_mentions = sum(cnt[c] for c in hard.canonical)
    rep = [f"Job có skills_raw: {n_jobs}/{len(df)}",
           f"Token: mapped(tổng lượt) {sum(var_cnt.values())}, noise {sum(noise_cnt.values())}, "
           f"tag Fresher/Intern đã bỏ {sum(drop_cnt.values())}, CHƯA map {sum(unmapped.values())} lượt / {len(unmapped)} token khác nhau",
           "", "Độ phủ HARD skill theo số cột giữ lại (xếp theo tần suất):",
           f"{'N':>5} {'min_jobs':>9} {'%mentions':>10} {'%jobs>=5 skill':>15} {'%jobs>=8 skill':>15}"]
    order = hard.sort_values("n_jobs", ascending=False).canonical.tolist()
    for n_top in (30, 50, 80, 100, 120, 150, 200):
        keep = set(order[:n_top])
        m = sum(cnt[c] for c in keep) / tot_mentions
        per_job = pd.Series({u: len(s & keep) for u, s in job_skills.items()})
        rep.append(f"{n_top:>5} {cnt[order[min(n_top, len(order)) - 1]]:>9} {100*m:>9.1f}% "
                   f"{100*(per_job[has.values] >= 5).mean():>14.1f}% {100*(per_job[has.values] >= 8).mean():>14.1f}%")
    for t in (5, 2, 1, 0.5):
        rep.append(f"df >= {t}%  -> {int((hard.df_pct >= t).sum())} hard, {int((soft.df_pct >= t).sum())} soft")
    is_global_s = sel.binary_reason.str.contains("global")
    is_role_core_s = sel.binary_reason.str.contains("role_core")
    rep += ["", f"in_binary: {len(hard_cols)} hard ({int(is_global_s.sum())} qua quy tắc (a) global >= {a.min_jobs} job, "
            f"{int(is_role_core_s.sum())} qua quy tắc (b) role_core, "
            f"{int((is_global_s & is_role_core_s).sum())} đạt CẢ HAI) + {len(lang_cols)} ngoại ngữ (>= {a.lang_min_jobs} job) + skill_cert_security (gộp {len(cert_members)} chứng chỉ)",
            f"Trong đó 'định danh nhóm nghề' (is_role_defining=True, tức đạt role_core ở >=1 nhóm): {int(hard.is_role_defining.sum())}/{len(hard)} "
            f"hard skill trong toàn bộ taxonomy, {int(sel[sel.type=='hard'].is_role_defining.sum())}/{len(hard_cols)} skill trong binary.",
            "", "=== Cơ cấu nhóm nghề trong dữ liệu (job có skills_raw) -- minh họa vì sao ngưỡng global (a) phụ thuộc cơ cấu này ===",
            "Nhóm nghề càng chiếm tỷ trọng lớn, skill của nhóm đó càng dễ đạt ngưỡng --min-jobs dù không đặc trưng cho nhóm; "
            "nhóm nhỏ (vd Data Scientist, Business Analyst) gần như CHỈ vào được binary nhờ role_core, không nhờ global."]
    for r_, n_r in sorted(role_tot_all.items(), key=lambda t: -t[1]):
        rep.append(f"  {r_:<18} n={n_r:>4}  ({100*n_r/n_has_total:>4.1f}% tổng job có skills_raw)")

    rep += ["", "=== Skill 'định danh' từng nhóm nghề (role_core, top theo lift) -- đây là core stack thật sự, khác với skill chung ==="]
    for r_ in roles:
        core_r = hard[hard.role_core_all.str.contains(re.escape(r_) + r" \(", regex=True, na=False)].copy()
        core_r["lift_r"] = core_r.role_core_all.str.extract(re.escape(r_) + r" \([\d.]+%, lift ([\d.]+|inf)\)")[0].replace("inf", "999").astype(float)
        core_r = core_r.sort_values("lift_r", ascending=False).head(10)
        rep.append(f"\n{r_} (n={role_tot_all.get(r_, 0)} job):")
        for rr in core_r.itertuples():
            rep.append(f"    {rr.canonical:<24} " + f"lift={rr.lift_r:>6.2f}  "
                       + (f"[CŨNG vào binary qua global -- skill phổ biến, không chỉ đặc trưng nhóm]" if "global" in rr.binary_reason else "[CHỈ vào binary nhờ role_core -- đặc trưng riêng của nhóm này]"))
        if core_r.empty:
            rep.append("    (không có skill nào đạt ngưỡng role_core cho nhóm này)")

    rep += ["", "=== Ghi chú rà soát: nhóm 'Security' có thể cần tách nhóm con trước khi diễn giải role_core ===",
            "Security gộp nhiều mảng khác nhau (SOC/vận hành, pentest/tấn công, GRC/tuân thủ, cloud security...). "
            "Danh sách dưới đây là các skill đạt role_core cho Security, kèm subgroup trong taxonomy -- nếu subgroup "
            "trải rộng (không dồn vào 1-2 subgroup), nên cân nhắc tách Security thành các nhóm con này rồi chạy lại "
            "role_core riêng cho từng nhóm con, thay vì gộp chung 1 nhóm 'Security' lớn:"]
    sec = hard[hard.role_core_all.str.contains(r"Security \(", regex=True, na=False)]
    if len(sec):
        for sg, g in sec.groupby("subgroup"):
            rep.append(f"  Subgroup '{sg}': " + ", ".join(g.canonical))
    else:
        rep.append("  (Không có nhóm nghề 'Security' trong dữ liệu hoặc không có skill nào đạt role_core cho nhóm này.)")

    langs_present = [l for l in LANGUAGES if cnt.get(l, 0) > 0]
    lang_drop = [l for l in langs_present if l not in lang_cols.values()]
    rep += ["", f"Ngoại ngữ có cột riêng (>= {a.lang_min_jobs} job): " + (", ".join(f"{l}({cnt[l]})" for l in lang_cols.values()) or "(không có)"),
            f"Ngoại ngữ dưới ngưỡng, không vào binary: " + (", ".join(f"{l}({cnt[l]})" for l in lang_drop) or "(không có)")]
    rep += ["", "Ngoại ngữ theo nhóm nghề (số job / tổng job của nhóm, từ JD text):"]
    for r_ in roles:
        ids = [u for u in df.job_uid[df.expertise_category == r_]]
        rep.append(f"  {r_:<15} n={len(ids):>4}  " + "  ".join(f"{l}={sum(l in job_skills[u] for u in ids)}" for l in langs_present)
                   + f"  >=1_ngoai_ngu={sum(any(l in job_skills[u] for l in LANGUAGES) for u in ids)}")
    rep += [
            "", "Soft skill (từ JD text) - % job theo nguồn:"]
    rep += [f"  {r.canonical:<24} all={r.df_pct:>5}%  " + "  ".join(f"{c[3:-4]}={getattr(r, c):>5}%" for c in tax.columns if c.startswith("df_") and c.endswith("_pct") and c != "df_pct")
            for r in soft.head(10).itertuples()]
    rep += ["", "(Xem thêm cột df_<nguồn>_pct trong skill_taxonomy.csv: hard skill lệch nguồn thường do cơ cấu vị trí (Xóm Jobs chủ yếu Data/AI), không phải lỗi trích xuất.)"]
    rep += ["", "Top 25 token chưa map: " + ", ".join(f"{t}({n})" for t, n in unmapped.most_common(25))]
    (out / "skill_report.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()