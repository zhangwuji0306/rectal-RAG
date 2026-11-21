# writing-rag

## Purpose

writing-rag 用于本机文献发现、全文规范化和检索。公共代码集用于验证管线、索引和评价逻辑。真实语料和生产运行状态由本机项目维护。

## Architecture

1. `直肠癌文献爬取/scripts/` 维护 canonical SQLite、官方 OA 候选、获取租约和正文校验。通过校验的正文可生成 normalized Markdown。
2. `index/` 提供源身份、分块、向量入库、账本校验和检索。现行生产分块为 passage-v2，默认 build 仍枚举 converted。
3. 检索结果可组装为 D1 Evidence Pack。D2 的正式答案由本地绑定证据和逐项人工确认放行。D3 的诊断与正式 final 独立计量。

## Quick Start

在按 [allowlist](docs/发布范围-allowlist.txt) 组装的干净代码目录使用 Windows 和 Python 3.12.0。GitHub Actions 使用同一 Python 版本及 `requirements.lock` 中的直接依赖版本。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock
.venv/Scripts/python.exe tests/run_public_tests.py
```

可用 `--check` 运行一个公共分组。分组、六个 required check 名称和目标仓库核验步骤见[发布清单](docs/数据合规与发布清单.md#d4-公共-ci-与目标仓库)。

## Corpus Policy

默认使用受支持的官方 OA 路径 `oa_resolver.py` 和 `fetch_fulltext.py`。来源、校验和许可证作为独立字段。legacy `03_downloader.py` 在配置及代码缺省中均关闭非 OA 路由，TLS 校验开启。历史兼容入口的使用须符合既有来源授权。真实正文、Chroma、模型、生产 SQLite、私有备份、凭据、Gold、人工标注和运行导出均留在本机。

## Retrieval

已有可用的本机索引、账本和模型配对后，从项目根运行 `.venv/Scripts/python.exe index/retrieve.py "<query>" -k 8`。生产通用基线沿用 D250 与 relevance-only，reranker disabled。更新或恢复目标依赖配对检查。检索命令会打开本机 Chroma，普通发布核查只需公共 fixture。

## Evaluation

公共检索回归使用合成排名与 qrels，schema 和索引测试使用临时数据库。真实评价按授权使用 `tests/run_local_acceptance.py` 及 [evaluation](evaluation/README.md) 的独立入口，记录 skip 和 not-run。正式答案沿用 D2 逐项人工确认，私有 Gold 和输出留在本机。

## Compliance

代码分发按显式 [allowlist](docs/发布范围-allowlist.txt) 隔离组装，用 `tests/release_scope.py` 扫描实际文件。当前本机 HEAD 及可达历史含非公开数据。GitHub 实跑、发布和分支保护须先核定目标仓库、权限及准确分发边界。见[数据合规与发布清单](docs/数据合规与发布清单.md)。

## Known limitations

D3 已按调整后的范围验收。正式 final 覆盖 1 道 development 题，直接冲突处理未实测，10 道 test 补评已取消。公共检索回归适用于合成 fixture。D4 本地补修待独立复审，GitHub Actions 实跑和 `main` 分支保护仍待在获授权的目标仓库验证。
