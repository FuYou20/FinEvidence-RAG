import json
import re
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "data/eval/retrieval_cases_v2.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data/eval/rag_benchmark_v1.jsonl"


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_evidence(source_cases, source_ids, evidence_indexes=None):
    evidence = []
    for source_id in source_ids:
        source = source_cases[source_id]
        selected = source["gold_evidence"]
        if evidence_indexes and source_id in evidence_indexes:
            selected = [
                selected[index] for index in evidence_indexes[source_id]
            ]
        for item in selected:
            copied = deepcopy(item)
            copied["source_case_id"] = source_id
            evidence.append(copied)
    return evidence


def build_answerable_case(source_cases, *, query_id, question, task_type,
                          subtype, difficulty, split, source_ids,
                          reference_answer=None, gold_items=None,
                          gold_facts=None,
                          evidence_indexes=None):
    evidence = source_evidence(
        source_cases,
        source_ids,
        evidence_indexes=evidence_indexes,
    )
    if reference_answer is None:
        reference_answer = source_cases[source_ids[0]]["reference_answer"]
    return {
        "query_id": query_id,
        "question": question,
        "task_type": task_type,
        "subtype": subtype,
        "difficulty": difficulty,
        "split": split,
        "answerable": True,
        "expected_action": "answer_with_citations",
        "requires_external_data": False,
        "evidence_structure": (
            "multi_document"
            if len({item["doc_id"] for item in evidence}) > 1
            else source_cases[source_ids[0]]["category"]
        ),
        "required_doc_ids": sorted({item["doc_id"] for item in evidence}),
        "gold_evidence": evidence,
        "gold_items": gold_items or [],
        "gold_facts": gold_facts or [],
        "reference_answer": reference_answer,
        "source_case_ids": source_ids,
        "annotation_status": (
            "composed_review_required"
            if task_type == "comparison"
            else "derived_from_human_reviewed"
        ),
    }


def build_unanswerable_case(*, query_id, question, subtype, difficulty,
                            split, expected_action,
                            requires_external_data, reason):
    return {
        "query_id": query_id,
        "question": question,
        "task_type": "unanswerable",
        "subtype": subtype,
        "difficulty": difficulty,
        "split": split,
        "answerable": False,
        "expected_action": expected_action,
        "requires_external_data": requires_external_data,
        "evidence_structure": "no_sufficient_evidence",
        "required_doc_ids": [],
        "gold_evidence": [],
        "gold_items": [],
        "gold_facts": [],
        "reference_answer": None,
        "unanswerable_reason": reason,
        "source_case_ids": [],
        "annotation_status": "review_required",
    }


def make_gold_items(source_cases, source_id, items, evidence_indexes=None):
    source_items = source_cases[source_id]["gold_evidence"]
    if evidence_indexes is not None:
        source_items = [source_items[index] for index in evidence_indexes]
    chunk_ids = [evidence["source_chunk_id"] for evidence in source_items]
    return [
        {"item": item, "evidence_chunk_ids": chunk_ids}
        for item in items
    ]


def build_cases(source_cases):
    cases = []

    single_specs = [
        ("传音控股旗下Boomplay在2020年第二季度的周人均打开次数和周人均使用时长分别是多少？", "ret_v2_003", "dev", "easy", [["21.6次"], ["40.78分钟"]]),
        ("TCL科技在2021年5月计划向TCL华星增资多少，增资后持股比例预计达到多少？", "ret_v2_006", "dev", "easy", [["50亿元", "50亿元人民币"], ["83.02%"]]),
        ("TCL科技2021年预期的半导体显示业务收入同比增长率是多少？", "ret_v2_008", "dev", "easy", [["110.73%"]]),
        ("TCL科技2022年的每股收益预测值是多少？", "ret_v2_013", "dev", "easy", [["0.80元", "0.8元"]]),
        ("华利集团2019年的鞋履总营收是多少亿元？", "ret_v2_019", "dev", "easy", [["151.66亿元", "151.66亿元人民币"]]),
        ("华利集团2021年第一季度至第三季度的归母净利润同比增速是多少？", "ret_v2_021", "dev", "easy", [["51.6%"]]),
        ("万科A成立于哪一年，又于哪一年实施股份制改革进入房地产行业？", "ret_v2_027", "dev", "medium", [["1984年"], ["1988年"]]),
        ("中顺洁柔2018年推出了什么定位中高端的产品？", "ret_v2_035", "dev", "easy", [["新棉初白"], ["棉柔巾"]]),
        ("安必平宫颈癌病变过程分期中，Ⅲ期患者的存活率是多少？", "ret_v2_044", "dev", "easy", [["40-50%", "40%至50%", "40%—50%"]]),
        ("广和通2019年的营业收入和净利润分别是多少，2015至2019年的复合增速是多少？", "ret_v2_023", "test", "medium", [["19.15亿元"], ["1.7亿元", "1.70亿元"], ["55.68%"], ["48.45%"]]),
        (
            "广和通2022年归属于母公司股东的净利润预计为多少？",
            "ret_v2_025",
            "test",
            "medium",
            [["564.15百万元", "564.15百万", "5.6415亿元", "约5.64亿元", "5.64亿元"]],
        ),
        ("招商银行2022年第一季度整体不良率环比提升多少个百分点，拨备覆盖率回落至多少？", "ret_v2_032", "test", "medium", [["3个基点", "3bp", "0.03个百分点"], ["462.7%"]]),
        ("广州酒家研报显示，2021年第二季度拥有1000家以上门店的餐饮企业占比是多少，较2017年提高多少个百分点？", "ret_v2_039", "test", "medium", [["14.5%"], ["6.6个百分点"]]),
        ("广州酒家其他食品业务2022年预测毛利率是多少？", "ret_v2_040", "test", "easy", [["26.5%"]]),
        ("比亚迪2018年9月签约建设的动力电池产能项目位于哪个城市，投资金额是多少？", "ret_v2_048", "test", "easy", [["西安", "西安市"], ["120亿元"]]),
    ]
    for offset, (question, source_id, split, difficulty, gold_facts) in enumerate(
        single_specs,
        start=1,
    ):
        cases.append(
            build_answerable_case(
                source_cases,
                query_id=f"rag_v1_{offset:03d}",
                question=question,
                task_type="single_fact",
                subtype="exact_fact",
                difficulty=difficulty,
                split=split,
                source_ids=[source_id],
                gold_facts=gold_facts,
            )
        )

    enumeration_specs = [
        ("华利集团合作的国际运动品牌有哪些？", "ret_v2_020", "dev", ["Nike", "Converse", "UGG", "Vans", "Puma", "Under Armour"]),
        ("华利集团2019年的产能分布在哪些地区，主要客户有哪些？", "ret_v2_022", "dev", ["越南", "缅甸", "印尼", "Nike", "Converse", "UGG", "Vans", "Puma", "Under Armour"]),
        ("中顺洁柔2019年完成了哪些主要研发项目和产品创新？", "ret_v2_036", "dev", ["洁柔系列再升级", "太阳系列研发", "商用消费领域个性化产品", "朵蕾蜜女性护理系列研发", "女性护理湿巾", "卸妆湿巾", "婴儿湿巾"]),
        ("中顺洁柔在2015至2020年间通过哪些方面的变革实现了业绩增长？", "ret_v2_038", "dev", ["稳定股权与管理", "完善产品矩阵", "拓展多元渠道", "布局全国产能", "持续研发创新", "人才与长期激励"]),
        ("安必平已经形成哪些病理诊断技术平台和产品线？", "ret_v2_046", "dev", ["细胞学诊断平台", "免疫诊断平台", "分子诊断平台", "LBP产品线", "PCR产品线", "FISH产品线", "IHC产品线"]),
        ("万科A从2004年至2018年经历了哪些战略定位变化？", "ret_v2_028", "dev", ["聚焦三大城市圈并追求有质量增长", "转型城市配套服务商", "升级为城乡建设与生活服务商"]),
        ("新宙邦在添加剂领域有哪些现有产能、建设规划和研发安排？", "ret_v2_016", "test", ["VC和FEC产能近5000吨", "收购淮安瀚康", "规划建设5.9万吨添加剂产能", "淮安翰康一期2.93万吨项目", "持续研发新型添加剂"]),
        ("新宙邦的电容器化学品包括哪些主要类别，相关电容器有哪些应用范围？", "ret_v2_018", "test", ["铝电解电容器化学品", "固态高分子电容器化学品", "超级电容器化学品", "噪声旁路", "电源滤波", "储能", "微分与积分电路", "振荡电路"]),
        ("广州酒家面临哪些主要风险因素？", "ret_v2_041", "test", ["高管减持风险", "所得税优惠政策取消风险", "食品安全风险"]),
        ("比亚迪向哪些车企外供动力电池，其产能规划提供了哪些支持？", "ret_v2_049", "test", ["北汽", "长安福特", "青岛索尔", "中通客车", "惠州基地已投产", "西安基地规划产能30并投资120亿元"]),
    ]
    for offset, (question, source_id, split, items) in enumerate(
        enumeration_specs,
        start=16,
    ):
        evidence_indexes = (
            {source_id: [0]} if source_id == "ret_v2_041" else None
        )
        cases.append(
            build_answerable_case(
                source_cases,
                query_id=f"rag_v1_{offset:03d}",
                question=question,
                task_type="enumeration",
                subtype="complete_list",
                difficulty="medium",
                split=split,
                source_ids=[source_id],
                gold_items=make_gold_items(
                    source_cases,
                    source_id,
                    items,
                    evidence_indexes=(
                        [0] if source_id == "ret_v2_041" else None
                    ),
                ),
                evidence_indexes=evidence_indexes,
            )
        )

    comparison_specs = [
        ("TCL科技2021年和2022年两份研报对公司业务重心及经营策略的描述发生了什么变化？", ["ret_v2_007", "ret_v2_011"], "dev", "cross_year_strategy", "2021年报告强调通过收购中环集团进入半导体和新能源领域，收入重心扩展到半导体显示、光伏及材料和分销业务；2022年报告面对面板价格下探，更强调优化产品结构、布局中小尺寸以及推动TV和商显高端化。"),
        ("TCL科技两份研报分别如何描述显示产能扩张和后续产品结构优化？", ["ret_v2_010", "ret_v2_014"], "dev", "cross_year_strategy", "较早报告强调收购苏州三星显示、茂佳国际和中环集团，并形成2条11代线、3条8.5代线及620K/月高世代产能；后续报告强调优化中小尺寸产品与客户结构，推进OLED、高附加值IT显示屏以及TV和商显高端化。"),
        ("TCL科技两份研报分别给出的2021年半导体显示业务收入预测和2022年每股收益预测是多少？", ["ret_v2_009", "ret_v2_013"], "dev", "cross_year_metric", "2021年报告预计半导体显示业务收入为985.49亿元；2022年报告预测每股收益为0.80元。两项数据来自不同报告和不同指标，不能直接作同比比较。"),
        ("TCL科技2021年报告对显示业务增长的预期，与2022年报告描述的面板价格压力有何差异？", ["ret_v2_008", "ret_v2_011"], "dev", "outlook_change", "2021年报告预计半导体显示业务收入同比增长110.73%；2022年报告则指出2021年第四季度受季节性淡季影响，面板价格持续下探，并提出产品结构优化和高端化策略。这反映了从增长预期转向价格压力应对。"),
        ("传音控股和华利集团的海外业务分别依赖怎样的市场机会、产能布局和客户基础？", ["ret_v2_002", "ret_v2_022"], "dev", "cross_company_strategy", "传音控股受益于非洲功能机向智能机升级及较大的智能机替代空间；华利集团则依靠越南、缅甸、印尼等地产能，以及Nike、Converse、UGG、Vans、Puma和Under Armour等国际客户。"),
        ("传音控股和中顺洁柔分别通过哪些路径推动增长？", ["ret_v2_005", "ret_v2_038"], "dev", "cross_company_strategy", "传音控股依靠新兴市场先发优势和移动互联网流量提升ARPU，并获得较高增长排名；中顺洁柔则通过稳定治理、产品升级、渠道协同、产能布局、研发和激励机制推动增长。"),
        ("华利集团与中顺洁柔在客户渠道和业务扩张方式上有什么不同？", ["ret_v2_020", "ret_v2_038"], "dev", "cross_company_strategy", "华利集团主要依赖Nike、Converse、UGG、Vans、Puma和Under Armour等国际品牌客户；中顺洁柔则通过电商、新零售、母婴、商销等多渠道协同，并扩展个护新品类和全国产能。"),
        ("万科A与TCL科技的业务多元化路径有何不同？", ["ret_v2_028", "ret_v2_007"], "dev", "cross_company_strategy", "万科从住宅供应商逐步转向城市配套服务商和城乡建设与生活服务商，拓展物流、商业、度假和养老；TCL科技通过收购中环集团进入半导体和新能源领域，形成显示、光伏及材料和分销等业务。"),
        ("万科A的偿债判断与TCL科技的资产扩张和产品策略分别说明了什么？", ["ret_v2_029", "ret_v2_014"], "dev", "cross_company_financial", "万科报告认为其杠杆率较低、短期偿债压力小，但流动负债预测规模上升；TCL报告预测2022年资产合计359,560百万元、流动资产194,352百万元，同时强调产品结构优化和显示业务高端化。二者指标口径不同，应分别解释。"),
        ("安必平和中顺洁柔分别如何通过研发与产品扩张增强竞争力？", ["ret_v2_046", "ret_v2_036"], "dev", "cross_company_innovation", "安必平形成细胞学、免疫和分子诊断平台及LBP、PCR、FISH、IHC产品线，并保持在研项目投入；中顺洁柔通过洁柔和太阳系列升级、个性化产品及女性护理产品研发扩充消费品矩阵。"),
        ("传音控股与安必平研报中的盈利预测和估值判断分别是什么？", ["ret_v2_005", "ret_v2_045"], "dev", "cross_company_valuation", "传音控股报告预计2022年收入615.08亿元、EPS 5.91元，对应PE约15倍；安必平报告预计2020年归母净利润90.09百万元，给予50至60倍PE和48.5至58.2元每股价格区间。两份估值对应的年份和公司不同。"),
        ("华利集团和TCL科技分别通过什么方式扩充生产能力与业务边界？", ["ret_v2_022", "ret_v2_010"], "dev", "cross_company_expansion", "华利集团通过越南、缅甸、印尼等地的海外产能服务国际运动品牌客户；TCL科技通过并购显示资产扩充面板产能，并收购中环集团进入光伏和半导体材料领域。"),
        ("广和通和比亚迪在汽车相关业务中的布局有何不同？", ["ret_v2_026", "ret_v2_049"], "test", "cross_company_strategy", "广和通依靠5G技术拓展车联网、车载信息服务和智能网络场景；比亚迪直接向北汽、长安福特、青岛索尔和中通客车外供动力电池，并通过惠州和西安产能规划支撑外供。"),
        ("新宙邦与比亚迪在新能源产业链中的扩产重点有何不同？", ["ret_v2_016", "ret_v2_048"], "test", "cross_company_expansion", "新宙邦重点扩充VC、FEC等电解液添加剂产能并规划5.9万吨项目；比亚迪在西安签约动力电池产能项目，投资120亿元。前者侧重电解液材料，后者侧重动力电池制造。"),
        ("招商银行2022年第一季度实际拨备覆盖率与报告预测的年末水平有何差异？", ["ret_v2_033"], "test", "within_document_comparison", "2022年第一季度实际拨备覆盖率为462.7%，报告预测2022年末为461.7%，预测值比一季度实际值低1.0个百分点。"),
        ("广州酒家和广和通研报分别给出了哪些盈利预测指标？", ["ret_v2_040", "ret_v2_025"], "test", "cross_company_metric", "广州酒家报告预测2022年毛利率为26.5%；广和通报告预测2025年归属于母公司股东的净利润为564.15百万元。两者指标、年份和单位不同，不能直接比较高低。"),
        ("广和通2020年前三季度的增长表现与比亚迪2020至2022年的盈利预期分别如何？", ["ret_v2_026", "ret_v2_050"], "test", "cross_company_financial", "广和通2020年前三季度收入增长48.01%至19.68亿元，净利润增长85.65%至2.24亿元；比亚迪2020至2022年净利润预计从2,119百万元增至8,364百万元，经营现金流预计维持在12,000百万元以上。"),
        ("新宙邦和比亚迪的产品分别覆盖哪些下游应用或客户？", ["ret_v2_018", "ret_v2_049"], "test", "cross_company_downstream", "新宙邦电容器化学品覆盖铝电解、固态高分子和超级电容器等类别，相关应用包括滤波、储能及振荡电路；比亚迪动力电池外供客户包括北汽、长安福特、青岛索尔和中通客车。"),
        ("广州酒家与招商银行研报分别指出了哪些可能影响盈利的压力因素？", ["ret_v2_041", "ret_v2_031"], "test", "cross_company_risk", "广州酒家面临高管减持、所得税优惠取消和食品安全等风险；招商银行2022年一季度财富管理中收同比下降8.7%，主要受代销公募基金和信托收入不及预期影响，但保险代销收入提供部分支撑。"),
        ("招商银行2022年一季度的收入压力与比亚迪2020至2022年的盈利预期呈现出怎样的差异？", ["ret_v2_031", "ret_v2_050"], "test", "cross_company_outlook", "招商银行财富管理中收同比下降8.7%，主要受基金和信托代销收入影响；比亚迪报告预计2020至2022年净利润从2,119百万元增至8,364百万元。前者描述短期收入压力，后者体现中期盈利增长预期。"),
    ]
    for offset, (question, source_ids, split, subtype, answer) in enumerate(
        comparison_specs,
        start=26,
    ):
        cases.append(
            build_answerable_case(
                source_cases,
                query_id=f"rag_v1_{offset:03d}",
                question=question,
                task_type="comparison",
                subtype=subtype,
                difficulty="hard",
                split=split,
                source_ids=source_ids,
                reference_answer=answer,
            )
        )

    unanswerable_specs = [
        ("TCL科技的A产品在2024年的市场反馈如何？", "ambiguous_entity", "dev", "clarify", False, "“A产品”无法唯一映射到语料中的具体产品。"),
        ("传音控股2026年第三季度手机出货量是多少？", "fresh_information", "dev", "external_research", True, "知识库中的传音研报发布于2022年，无法支持2026年数据。"),
        ("万科A在2024年新进入了哪些海外城市？", "out_of_corpus", "dev", "abstain", False, "现有万科研报没有该时间范围和事实。"),
        ("比亚迪内部代号B7电池尚未公开的良率是多少？", "private_information", "test", "abstain", False, "问题要求未公开内部信息，语料没有可靠证据。"),
        ("广州酒家2025年全年实际净利润是多少？", "fresh_information", "test", "external_research", True, "现有研报发布于2022年，无法提供2025年实际全年数据。"),
    ]
    for offset, (question, subtype, split, action, external, reason) in enumerate(
        unanswerable_specs,
        start=46,
    ):
        cases.append(
            build_unanswerable_case(
                query_id=f"rag_v1_{offset:03d}",
                question=question,
                subtype=subtype,
                difficulty="medium",
                split=split,
                expected_action=action,
                requires_external_data=external,
                reason=reason,
            )
        )

    return cases


def validate_cases(cases):
    expected_counts = {
        "single_fact": 15,
        "enumeration": 10,
        "comparison": 20,
        "unanswerable": 5,
    }
    if len(cases) != 50:
        raise ValueError(f"题目数量必须为50，实际为{len(cases)}")
    if len({case["query_id"] for case in cases}) != len(cases):
        raise ValueError("query_id 必须唯一")
    actual_counts = {
        task_type: sum(case["task_type"] == task_type for case in cases)
        for task_type in expected_counts
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"任务类型数量错误: expected={expected_counts}, actual={actual_counts}"
        )
    split_counts = {
        split: sum(case["split"] == split for case in cases)
        for split in ("dev", "test")
    }
    if split_counts != {"dev": 30, "test": 20}:
        raise ValueError(f"切分数量错误: {split_counts}")

    location_hint = re.compile(r"第\s*\d+\s*页|根据(?:表|图表)\s*\d+")
    for case in cases:
        if location_hint.search(case["question"]):
            raise ValueError(
                f"问题不得泄露证据位置: {case['query_id']} {case['question']}"
            )
        evidence_doc_ids = sorted(
            {item["doc_id"] for item in case["gold_evidence"]}
        )
        if evidence_doc_ids != case["required_doc_ids"]:
            raise ValueError(f"required_doc_ids 不匹配: {case['query_id']}")
        if case["answerable"]:
            if not case["gold_evidence"] or not case["reference_answer"]:
                raise ValueError(f"可回答问题缺少标注: {case['query_id']}")
        elif case["gold_evidence"] or case["reference_answer"] is not None:
            raise ValueError(f"不可回答问题不应有答案证据: {case['query_id']}")
        if case["task_type"] == "single_fact" and not case["gold_facts"]:
            raise ValueError(
                f"single_fact 缺少可机器校验的 gold_facts: {case['query_id']}"
            )
        if case["task_type"] == "enumeration" and not case["gold_items"]:
            raise ValueError(
                f"enumeration 缺少可机器校验的 gold_items: {case['query_id']}"
            )
        available_chunk_ids = {
            item["source_chunk_id"] for item in case["gold_evidence"]
        }
        for gold_item in case["gold_items"]:
            unknown_chunk_ids = (
                set(gold_item["evidence_chunk_ids"]) - available_chunk_ids
            )
            if unknown_chunk_ids:
                raise ValueError(
                    f"gold_items 引用了本题之外的证据: "
                    f"{case['query_id']} {sorted(unknown_chunk_ids)}"
                )

    split_docs = {}
    for split in ("dev", "test"):
        split_docs[split] = {
            doc_id
            for case in cases
            if case["split"] == split
            for doc_id in case["required_doc_ids"]
        }
    overlap = split_docs["dev"] & split_docs["test"]
    if overlap:
        raise ValueError(f"dev/test 文档发生重叠: {sorted(overlap)}")
    return {
        "total": len(cases),
        "task_counts": actual_counts,
        "split_counts": split_counts,
        "dev_doc_count": len(split_docs["dev"]),
        "test_doc_count": len(split_docs["test"]),
    }


def main():
    source_rows = load_jsonl(SOURCE_PATH)
    source_cases = {row["query_id"]: row for row in source_rows}
    cases = build_cases(source_cases)
    summary = validate_cases(cases)
    OUTPUT_PATH.write_text(
        "".join(
            json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
