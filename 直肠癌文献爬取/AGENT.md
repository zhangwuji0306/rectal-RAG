# AGENT.md — 直肠癌文献爬取项目

更新：2026-09-30。本目录受项目根 `AGENTS.md` 约束。当前问题见根目录 `ABCD整改审查意见.md` 第 3 节，P05 收口合同见第 5 节；阶段状态见 `整改手册/Status.md`。

## 权威数据与范围

`tasks.sqlite` 是 metadata、业务状态和来源记录的权威源；CSV、reports 和语料元数据是导出物，不反向覆盖当前 SQLite。原 NBIB 保留不变。范围沿用 2000 年及以后，archived 记录不得通过手工 SQL 批量复活。

正文目标为可信、经过验证的 XML/PDF/TXT/HTML，再生成 normalized 文档；当前各格式支持程度不同，不能把下载成功当作可入库。`pdfs_merged/` 是历史 PDF 主副本，不因下游转换删除。

## 已有模块与实际边界

| 入口 | 作用及限制 |
|---|---|
| `ingest_pmids.py` / `a1_schema.py` | canonical schema 与 discovery，不以 CSV 覆盖业务状态。 |
| `refresh_pubmed_metadata.py` | 单个/列表/明确全库 refresh；功能实现不代表全库已刷新。 |
| `oa_resolver.py` | dry-run 仍写 metadata/candidates/status，只是不下载正文；PMC AWS 需本地 inventory。 |
| `fetch_fulltext.py` | 单篇 raw 获取；领取 lease、请求前后续租及所有权校验、A6 失败 fallback 已实现；已有验证正文默认跳过，仅显式 refresh 重新获取。A-R2/A-R3 已按 P01/P02 当前支持边界验收。 |
| `content_validation.py` | XML/PDF 双验证；TXT/HTML 当前未支持。A-R1 绑定列（`validated_content_path`/`validated_content_sha256`）已随加字段迁移落到 canonical 库。 |
| `normalization.py` | 生成 normalized Markdown；A7 只解析 A6 已核验的那一份字节（A-R1 绑定）。PDF MinerU API adapter 不是离线工具，不得擅自外传正文。 |
| `lease_batch.py` | claim/heartbeat/complete/reclaim；只回收实际过期 lease，不按 updated_at 推断；通用写入口的 owner/有效 lease 约束已按 A-R2 验收，失效 worker 的写入被拒绝。 |
| `a9_reporting.py` | canonical/legacy 联合报告；gate 为本次检查结果、readiness 分调度状态与正文就绪两个维度（A-R4），`--json-out` 出机器可读报表（真实报表见 `reports/a9-unified-report-*.json`）。 |
| `03_downloader.py`、`11/12/13` 系列 | legacy 获取与批次兼容工具，非默认下一步，仅用于明确维护/迁移任务。 |

A8 已有 canonical batches/batch_members，保留历史来源关系。B3 无 batch 只针对索引重建，不删除 acquisition 分组。旧批次库合并保留恢复副本，不按“随跑随删”操作。

`normalized/` 尚未接入默认 build_index source 枚举；index/ingest_normalized.py 已完成隔离入库及重跑集合核验，不等同 normalized→生产 index 已验收。不手工覆盖 converted 文件。全量 A rollout 未完成，A9 的 normalized indexed 与 index consistency 保持 NOT CHECKED。

P04 已获 ACCEPT_WITH_FINDINGS，生产 DB/manifest 已恢复为 passage-v2 配对：79,239 rows/chunks、2,492 canonical sources、2,499 manifest entries。B3 指标下降及 PersistentClient 查询改变 SQLite SHA 的 finding 保留，具体见整改手册/Status.md；P06/B4 暂不放行。禁止以 reconcile 写模式将丢失向量合理化，不按历史 partial 快照重建，不为状态核对重查生产 Chroma。

## 历史磁盘数据库快照（2026-09-22）

2026-09-22：9,171 tasks；done 2,419 / pending 4,073 / not_found 1,075 / failed 21 / metadata_only 1 / archived 1,582。metadata_updated_at 非空为 0；source_candidates 1 / fetch_attempts 1；一个任务 content_status=fulltext_ready 且双验证通过。读取方法与限制见根审查意见，不代表已完成全库在线刷新。

A-R1 加字段迁移已于 2026-09-22 在隔离副本验证后对 canonical `tasks.sqlite` 执行：新增 `validated_content_path`/`validated_content_sha256`，9171 行均未回填，无行数据变化。唯一的历史验证行（PMID 31096736）无绑定列记录，按 A5 `content_sha256` 回退判定绑定，并已按字节复核其 path 与 sha256 与磁盘一致；没有需要记待验证的旧行。

历史 PDF 与批次统计见既有 reports，旧 summary.txt 不是当前权威状态。来源成功、双验证成功和入库成功分别统计。

## 执行约定

从项目根使用 `.venv/Scripts/python.exe`，设置 `PYTHONIOENCODING=utf-8`。网络获取、迁移和批次工具按当前任务明确范围执行。

令牌和联系参数取进程环境 `NCBI_API_KEY`、`UNPAYWALL_EMAIL`、`CROSSREF_MAILTO`，不写配置文件或报告。默认只用已批准官方 OA 来源并保持 TLS 校验；既有非 OA 来源审批边界不因文档更新放宽。

测试使用合成 fixture 和临时 SQLite。raw/normalized/converted、生产数据库和备份不进入公开发布物；D-R4 已取消 raw XML 等禁止内容的 Git 索引跟踪，磁盘原件保留；HEAD/可达历史仍含禁止发布内容。普通代码分发按 docs/发布范围-allowlist.txt 从工作树组装；本地 tracking ref 不是实时远端证据。取消跟踪不等于清理历史，不删除本地原件。
