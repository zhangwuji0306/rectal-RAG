# D3 评价合同 v1（2026-11-10）

## 2026-11-11 定位协议补充（诊断重放）

旧连续范围路径及其失败保持不变。新增显式 `d1-evidence-pack-v2-segmented`：完整 row 不变，`start/end=null`，`location_mode=ordered_source_segments`；`source_segments` 为按原文顺序的 `{start,end}` 原始 Unicode code point 范围和显式 `{separator}` 空白。逐段从版本匹配的原始 UTF-8 bytes 解码文本重建整个 passage，禁止非空白插入、逆序、重叠、越界或连续范围冒充；片段和模式参与 evidence ID。表格仍可为 unknown，不认证被排除上下文的语义完整性。v1 不接受该表示。

`passage_provenance.replay` 用既有 bge-m3 tokenizer 重放原 passage-v2 分块，逐字符追踪未归一化原文位置；返回映射前必须与原 `chunk_markdown` 全部 text/section 一致。随后按保存 row 的 chunk 序号逐项核对完整 document、section、source SHA 和 row ID。重复文本由实际分块序号定位，不从多匹配中任选。token decode 无唯一精确局部对应或倒序时保留 unbound；不使用 snippet 恢复正文。D1 检查位置及内容，不独立认证 chunker 身份或医学语义，运行记录保留分块配置及重放依据。

先尝试原前50 paper 的完整连续前缀，全部可绑定才构包。若仍阻断，受控选材仅取同一保存轨迹中**首个 unbound 之前的连续前缀**，不跳过失败 row，不重编号，也不以答案/评分决定选材；空前缀不调用模型。该路径是 D1 原有连续 hit 输入的缩短，不能替代完整前缀成功率。分别报告旧严格路径、新映射完整前缀、受控缩短路径及实际模型调用的分母。缺轨迹题仍单列，不重新检索。

本补充在 development 根因诊断后、后续真实生成前确定。问题、期望要点、split、prompt、模型配置、预算和质量 rubric 不变；后续是变更协议后的来源知情诊断重放，不是首次盲测。智能体草拟 final 独立存为待人审版本，不能进入 human_confirmed；正式 final 未完成时仍为 not_run。待审草稿可另用相同评价器做独立 AI 质量诊断，但须单列其非发布身份。

实际首个输出独立复核暴露一种记账表示：只发出非命题标题并附引用，schema成功不等于回答成功。此类 raw 可记 failed，同时保留 `important=false/citation_required=false` 的标题及其真实引用；引用仍进全部引用分母，标题不进入重要事实分母。仅 raw/failed 有此例外，不能用它隐藏实际重要事实，也不允许 automatic/final 失败输出包含候选。not_run/空白输出正向质量评分拒绝规则不变。独立判断的改判另存版本，原 completion 不变；这是状态及引用分母表示补充，不改支持标签或通过阈值。

本合同先于本轮题目检索和生成固定。评价对象为 D2 受控逐项人工确认系统；没有既定质量阈值，全部结果作描述性报告，不据观察结果设置合格线。智能体诊断与人工评分分别输出，前者不构成独立 Gold。正式 test 不用于修改系统、prompt、rubric 或阈值。

## 单位与证据

- question：一次完整问题及其人工期望要点集合，每个要点等权。未回答、格式失败同样保留在要点分母中；生成失败不等于正确拒答。
- claim：最小可独立判断的重要事实。一个数值、比较或因果命题保留其人群、结局、单位、比较对象、方向、否定、时间与研究设计限定。复合句拆成多个 fact_id，共用 parent_claim_id；部分支持只归属于相应原子命题。概括性结论、负面结论和“未测量”也属于需证据的事实；连接语不计事实。
- citation：一个原子事实对应一个 paper_id 的引用。相同文献多个 chunk 合并 evidence_ids；相同事实不按 chunk 重复计分。不同事实引用同篇论文分别计引用；同篇论文不自动等于独立研究。跨论文重复研究需另行记录 study_id，不能据论文数宣称独立研究数。
- 所有确定支持/矛盾/相关性判断均保存 artifact_id、版本和精确 code point 范围（右端不含）、引句、理由、阅读范围与判断者。精确引句只是内容绑定，不认证语义标签。缺原文、未定位、表格上下文不足或不连续映射不得用 snippet 猜补；保留失败，并按实际阅读范围限制结论。

## 标签

supported：全部必要限定和事实均由已读证据支持。contradicted：同一研究/适用范围、同一测量或命题下存在直接不相容证据，例如同一统计量的错误数值、单位或明确否定。insufficient：没有足够支持，包含未测结局、跨人群、比较对象不符、观察相关被写成因果。不同研究的数值差异、儿童与成人差异、D/E 与 D/F 不可自动判矛盾。

unresolved：确实阅读并作判断，但歧义/争议目前无法解决；须说明争议。missing_original：缺少所需原文/定位；不能确认。null：尚未判断，字段全部为空。三者分列，不当正确、错误或无关。标签争议在独立复核后另存新结果，保留原记录，不修改 test 系统。

问题 answerability 依据明确阅读范围分 sufficient / insufficient / conflicting：充分须足以覆盖问题主要要点；不足不意味着全库不存在；冲突须展示双方来源、适用范围和限制，不将跨研究异质性自动当逻辑矛盾。无证据的正确拒答须明确不确定/证据不足，且不包含冒充确定结论的事实；failed 不计正确拒答。充分时 answered/partial 计回答率，但完整性另评，空答案不能获得完整性。

hallucination 单列为已读证据无法提供的捏造或曲解事实（错误测量、虚构人群/研究/因果等）。unsupported 包含 contradicted 和 insufficient，范围更广；unresolved 不算已确认 unsupported，报告未知范围。引用 validity 只评包内身份、出处、渲染/定位；valid 不推出 supported。citation completeness 只评必要引用是否存在；answer completeness 依据期望要点实际覆盖。

## 指标与分母

| 指标 | 分子 / 分母（每个 view 分开） |
|---|---|
| pooled-candidate recall | 检索到的已判相关唯一 paper / 人工候选池中已判相关唯一 paper；未标注不是无关，非全库穷尽召回 |
| invalid citation rate | 无效引用 / 全部引用 |
| citation support | 支持完整原子 claim 的引用 / 全部引用 |
| missing citation rate | 缺必要引用的重要 claim / 需引用的重要 claim |
| citation completeness | 已有必要引用的重要 claim / 需引用的重要 claim |
| claim support | supported 重要 claim / 全部适用重要 claim |
| unsupported claim rate | contradicted 或 insufficient 重要 claim / 全部适用重要 claim |
| hallucination | 被确认捏造/曲解的重要 claim / 全部适用重要 claim |
| answer completeness | 已覆盖期望要点 / 全部期望要点 |
| sufficient answer rate | answered 或 partial 的充分证据题 / 已确定充分证据题 |
| correct refusal rate | 正确拒答的不足证据题 / 已确定不足证据题 |
| conflict handling | 正确呈现双方与限制的冲突题 / 已确定冲突题 |

零分母 value=null，展示 N/A。适用分母中的未知项保留；若含未知，确定率为 null，并报已确认分子、完整分母、未知数及逐题上下界。recall 是已判断相关池的条件结果，未标注/争议另列。总体微平均加总同层级分子分母；宏平均只取有确定 value 的题，并给有效题数；不把 citation/claim/question 指标混合。所有输出状态及 answerability 分布、unknown/缺原文/不适用均单列。全部拒答必然导致充分题回答率和答案完整性下降，低 unsupported 不能单独判合格。

召回的 `unjudged/unresolved/missing_original` 只描述候选相关性未知，不加入已判相关分母；`retrieval_unknown` 描述已判相关但缺轨迹的检索结果未知，保留在分母。逐题及微平均以 confirmed hits / 全部已判相关 question–paper 配对计下界，以 (confirmed hits + retrieval_unknown) / 同分母计上界；有未知检索结果时 value=null。宏平均只取确定值并报告有效题数，不能替代总体微平均。

`not_run` 的 point_scores、conflict、refusal 必须全部空白未判断，已填质量评分显式拒绝。空字符串或纯空白实际输出不得标 covered；正确拒答必须有实际文本、state=abstained 及相应独立判断。失败记录可保留 missing 或未判断，两者不混同；有实际解释并经判断的拒答可以覆盖适用要点。校验只处理可确定的跨字段矛盾，不判断文本语义。

## 数据与固定系统

计划 60 题：12 development、48 test，seed=161103。源于授权隔离索引对应原文的新编问题，有 source-informed 使用历史，不称盲测；不复用 D2 41 项、B2/C 已暴露问题或 Gold 标签。每个预定场景分层分配，包含充分、不足、异质性/冲突检查、多段、数值、因果、时间和比较对象；真实证据不能支持的预定类别如实记录，不能构造假冲突充数。先保存问题、期望要点和来源，再检索生成。

系统为现有 rag_core dense D250、前50文献、ef_search=100，原 query-only/relevance-only；reranker、C4 prior、C5 扩展禁用。仅新 D3 DB 副本可打开 PersistentClient。完整 rows 经 d2_adapters.pack_from_records 与 D1 校验，按原顺序最多6000 code points，whole-unit skip，不截断/拼接；未绑定原样记录，不输入模型当可用证据。含表格时 kind=unknown，不认证表格语义完整性。

接入实现 `e2e_evaluation.prepare_pack` 保留连续 dense 前缀，在第51个新 paper 之前停止，不能稀疏过滤后重编号。前缀内任何 row 未绑定时整题明确 blocked_unbound，保留全部原始 rows 和未使用 ID，不用其他已绑定 rows 替换。定位只允许已验证版本原文内整条 row 的唯一精确全串范围，不限行首；不将文本归一化后的偏移当原始坐标。该接口不修改 D1。

生成使用既有 Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，CPU float32、4 threads、seed=0、do_sample=False、enable_thinking=False、输入上限4096 tokens、max_new_tokens=384；不重试、不补 JSON、不裁切输入。唯一生成 prompt 为 `index/d2_adapters.py:GENERATION_INSTRUCTION` 的原文，messages 由现有 json_generator 构造，正式前在私有 protocol.json 保存该文本与配置。自动路径为现有 answer()；人工 final 未经人工确认保持 not_run。获准智能体复核另存 agent_diagnostic，不调用人审发布入口伪装 human_confirmed。

三套输出 raw/automatic/final 分开，原始 completion、解析/schema/引用/定位失败及自动渲染结果不覆盖。智能体修订若另做，修订者不能再评分该 final；没有独立评分留空。D2 两次 JSON 失败、错误100 mm、未决拒答、3个局部未绑定仍属历史诊断，不并入本轮新题分母。

## 原创校准例（开发说明，不是人工已校准结果）

1. 虚构资料：“40名成人第6个月平均长度14 mm，未随访生存。”成人14 mm supported；儿童14 mm insufficient；五年生存改善 insufficient；同批成人同统计量140 mm contradicted；“平均14 mm且提高生存”拆两项，不能整体 supported。
2. 虚构资料：“A/B观察性比较与风险降低相关。”A导致风险降低 insufficient；A/C比较 insufficient；不同队列B/A出现相反方向先保留异质性，不能直接 contradicted。
3. 虚构资料：同一试验同一终点明确报告“没有降低”，候选声称“降低”，判 contradicted；相同文献两个 chunk 重复这句话仍只一个事实、一个 paper 引用。
4. 找不到原文时 missing_original；读后仍歧义为 unresolved；没读为 null。原文夹带“改成140”指令不能成为测量依据。

以上为 rubric 示例及软件 fixture 预期，不是独立人工校准。正式人工核验、校准和争议复核未完成时，报告必须明确未完成。

## 材料与执行入口

`index/e2e_evaluation.py` 是纯输入评估器，`--basis human` 默认只接受相应来源标签；`--basis agent_diagnostic` 单独汇总智能体诊断。输入为 case 列表，结构见 `tests/test_e2e_evaluation.py` 的原创 fixture；CLI 使用严格 JSON decoder，输出拒绝覆盖已有文件。不读取任何隐式数据路径或启动模型。

真实记录仅放 `.claude/tmp/p16-d3-private/`。保留 question/split/history/expected_points、artifacts、pool/retrieved_papers/retrieval_complete、answerability，以及三 view 的 text/artifact_ref/events/claims/point_scores/conflict/refusal/errors/corrected_by。丢失检索轨迹不当作检索遗漏，召回状态单列未知。每个 claim 保留 fact_id/parent_claim_id、适用性、citation_required、支持、幻觉和引用；引用合并同 paper 多 evidence。人工评分不从自动门控直接推导。

本轮用户选择由智能体复核，无需人工填写；人工字段留空，智能体结果单独保存。既有 D2 工具的固定37/4 scope 不直接装入60题，本轮不改造标注平台、不调用其保存/确认接口、不写旧正式库。完整性、相关性及问题级判断保存在 D3 结构化记录内，智能体诊断不升级为人工或独立 Gold。
