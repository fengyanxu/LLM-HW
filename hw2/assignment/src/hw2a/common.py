"""Shared paths, IO helpers and PeopleDaily token parsing."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _path in (DATA, MODELS, RESULTS, FIGURES):
    _path.mkdir(parents=True, exist_ok=True)

PEOPLEDAILY_DIR = Path(r"D:\Projects\LLM-HW\hw1\data\raw\199801")
ID_RE = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-\d{3}/m$")
SENT_END = re.compile(r"(?<=[。！？!?])")
TOKEN_TAIL = re.compile(r"(?:\][A-Za-z]+)+$")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_token(raw: str) -> str | None:
    token = raw.lstrip("[")
    token = TOKEN_TAIL.sub("", token)
    if "/" not in token:
        return None
    word = token.split("/", 1)[0].strip()
    if not word:
        return None
    if any(unicodedata.category(char).startswith("C") for char in word):
        return None
    return word


def split_sentences(tokens: list[str]) -> list[list[str]]:
    sentences: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        current.append(token)
        if SENT_END.search(token):
            if len(current) >= 3:
                sentences.append(current)
            current = []
    if len(current) >= 3:
        sentences.append(current)
    return sentences


def clean_title(words: list[str]) -> str:
    title = "".join(words)
    return title.strip("　 \t\r\n，。、；：！？“”‘’《》（）()[]【】—…·")


# 标题关键词只用于“挑候选文章”，不参与分类器输入。
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "体育": [
        "足球", "篮球", "排球", "乒乓球", "羽毛球", "网球", "棒球", "垒球", "手球",
        "比赛", "冠军", "亚军", "运动员", "教练", "联赛", "奥运会", "亚运会",
        "世界杯", "体育", "赛场", "进球", "球队", "田径", "游泳", "体操", "棋",
        "桥牌", "滑雪", "滑冰", "射击", "举重", "拳击", "柔道", "跆拳道",
        "马拉松", "赛车", "球迷", "国家队", "集训", "预选赛", "锦标赛", "运动员",
    ],
    "金融": [
        "银行", "贷款", "利率", "存款", "储蓄", "金融", "证券", "股票", "股市",
        "保险", "货币", "汇率", "投资", "基金", "债券", "期货", "信贷", "财险",
        "人寿", "信用社", "外汇", "通货膨胀", "央行", "人民银行", "工商银行",
        "建设银行", "农业银行", "中国银行", "交通银行", "招商银行", "融资",
        "资本市场", "上市公司", "股民", "利息", "金融监管", "金融风险",
        "工行", "建行", "农行", "中行", "交行", "招行", "行长", "储户",
        "债市", "汇市", "保险业", "证券业", "金融业", "金融界", "信用社",
        "合作银行", "政策性银行", "商业银行", "投资银行", "证券交易",
    ],
    "教育": [
        "教育", "学校", "学生", "教师", "老师", "高考", "中考", "大学", "中学",
        "小学", "招生", "课程", "教学", "校园", "幼儿园", "研究生", "博士",
        "留学", "考试", "教材", "校长", "课堂", "学费", "毕业", "教育部",
        "义务教育", "素质教育", "高校", "学院", "系主任", "班主任", "考生",
    ],
    "文化": [
        "文化", "艺术", "文物", "博物馆", "演出", "电影", "戏剧", "京剧", "音乐",
        "舞蹈", "图书", "出版", "文学", "作家", "画家", "书法", "摄影", "展览",
        "节目", "电视剧", "演员", "剧院", "乐团", "文化遗产", "考古", "戏曲",
        "话剧", "音乐会", "美术馆", "出版社", "小说", "诗歌", "文艺", "导演",
    ],
    "科技": [
        "科技", "技术", "科研", "科学家", "计算机", "网络", "信息", "软件",
        "硬件", "互联网", "发明", "专利", "卫星", "航天", "生物", "基因",
        "通信", "电信", "电子", "自动化", "新材料", "能源", "高技术", "高新技术",
        "科技成果", "科技攻关", "信息技术", "科技园", "研究所", "科学院",
        "工程院", "院士", "科技创新",
    ],
    "医疗": [
        "医疗", "医院", "医生", "大夫", "卫生", "疾病", "药品", "药物", "健康",
        "患者", "病人", "手术", "防疫", "疫苗", "中医", "西医", "护士", "诊所",
        "癌症", "艾滋病", "传染病", "医保", "医药", "药店", "门诊", "急诊",
        "住院", "体检", "血吸虫", "地方病", "妇幼", "卫生防疫", "医学",
        "防治", "病情", "就诊", "医护人员", "卫生院", "急救", "肿瘤", "糖尿病",
        "高血压", "精神病", "结核", "肝炎", "献血", "输血", "临床", "口腔",
        "眼科", "儿科", "妇科", "骨科", "心脏", "肾脏", "康复", "保健",
        "救治", "抢救", "中毒", "伤亡", "看病", "治病",
        "治疗", "诊断", "病房", "中医药", "中药", "药厂", "医疗队", "卫生所",
        "红十字", "血癌", "医务人员", "医术", "疗效", "病症", "发病率",
        "死亡率", "预防", "义诊", "送医", "医药费", "药品价格", "卫生院",
        "医疗费", "公费医疗", "合作医疗", "儿童", "孕产妇", "残疾人",
    ],
    "农业": [
        "农业", "农村", "农民", "粮食", "水稻", "小麦", "玉米", "棉花", "种植",
        "养殖", "丰收", "耕地", "农田", "水利", "化肥", "农药", "种子", "畜牧",
        "渔业", "林业", "乡镇企业", "菜篮子", "夏粮", "秋粮", "春耕", "抗旱",
        "防汛", "农业科技", "农业产业化", "粮价", "棉农", "果农", "蔬菜",
        "农技", "田头", "麦田", "稻田", "菜田", "果园", "饲料", "农机",
        "农资", "粮站", "棉区", "烟叶", "茶叶", "糖料", "油料", "猪",
        "牛羊", "家禽", "鱼塘", "畜牧", "兽医", "农垦", "垦区", "吨粮田",
    ],
    "法治": [
        "法律", "法院", "法官", "检察", "公安", "犯罪", "案件", "判决", "律师",
        "立法", "执法", "司法", "宪法", "刑法", "民法", "诉讼", "被告", "原告",
        "监狱", "警察", "诈骗", "贪污", "受贿", "盗窃", "杀人", "纠纷", "仲裁",
        "法制", "法治", "依法", "违法", "犯罪嫌疑人", "法庭", "起诉", "审判",
    ],
}
