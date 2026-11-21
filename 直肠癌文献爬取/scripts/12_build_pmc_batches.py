# -*- coding: utf-8 -*-
"""阶段 12：安全构建 PMC 优先批次队列。

所有成员、来源和既有 canonical batch 一致性先在内存与临时 SQLite 中校验。
校验成功后才登记 canonical membership，并原子安装批次快照与 manifest；失败
时不触碰既有快照或清单，成功替换的旧文件也保留为可恢复的 previous 文件。
"""

import argparse
import csv
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, connect_db, now_str, norm_text
from lease_batch import ensure_lease_schema

BATCH_NOTE = 'PMC priority batch selection'


def _load_pool(src):
    conn = sqlite3.connect(src)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT pmid, doi, pmc, year, title, status, license FROM tasks "
            "WHERE year>=2000 AND status!='archived' AND pmc IS NOT NULL AND pmc!='' "
            "AND status IN ('pending','not_found','failed') "
            "ORDER BY year DESC, pmid DESC"
        ).fetchall()
    finally:
        conn.close()


def _plans(pool, batch_size, max_batches, first_batch):
    batches = [pool[i:i + batch_size] for i in range(0, len(pool), batch_size)]
    if max_batches:
        batches = batches[:max_batches]
    return [
        (f'pmc_b{i}', list(batch))
        for i, batch in enumerate(batches, first_batch)
    ]


def _ensure_canonical(src):
    ensure_lease_schema(src)


def _validate_existing_batches(src, plans):
    """Validate all canonical provenance and membership before any output write."""
    _ensure_canonical(src)
    conn = sqlite3.connect(src)
    conn.row_factory = sqlite3.Row
    try:
        for batch_id, batch in plans:
            expected = {int(row['pmid']) for row in batch}
            existing = conn.execute(
                'SELECT source, query_text, note FROM batches WHERE batch_id=?',
                (batch_id,),
            ).fetchone()
            if existing is None:
                continue
            if (existing['source'], existing['query_text'], existing['note']) != (
                'pmc', None, BATCH_NOTE
            ):
                raise ValueError(f'canonical provenance mismatch: {batch_id}')
            actual = {
                int(row[0]) for row in conn.execute(
                    'SELECT pmid FROM batch_members WHERE batch_id=?', (batch_id,)
                ).fetchall()
            }
            if actual != expected:
                raise ValueError(
                    f'canonical membership mismatch for {batch_id}: '
                    f'expected {sorted(expected)}, '
                    f'got {sorted(actual)}'
                )
    finally:
        conn.close()


def _stage_batch_db(path, batch):
    conn = connect_db(path)
    try:
        for row in batch:
            conn.execute(
                'INSERT INTO tasks (pmid, doi, pmc, year, title, title_norm, status, attempts, license, updated_at) '
                'VALUES (?,?,?,?,?,?,?,?,?,?)',
                (row['pmid'], row['doi'], row['pmc'], row['year'], row['title'],
                 norm_text(row['title']), 'pending', 0,
                 row['license'] or 'unverified', now_str()),
            )
        conn.commit()
    finally:
        conn.close()
    for suffix in ('-wal', '-shm'):
        sidecar = str(path) + suffix
        if os.path.exists(sidecar):
            raise RuntimeError(f'staged SQLite sidecar remains: {sidecar}')


def _stage_manifest(path, plans):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Batch', 'PMID', 'Year', 'DOI', 'PMC', 'Status', 'Title', 'License'])
        for batch_id, batch in plans:
            for row in batch:
                writer.writerow([
                    batch_id, row['pmid'], row['year'], row['doi'], row['pmc'],
                    row['status'], row['title'], row['license'] or 'unverified',
                ])


def _validate_staged_db(path, batch):
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute('SELECT pmid, status FROM tasks ORDER BY pmid').fetchall()
    finally:
        conn.close()
    expected = sorted(int(row['pmid']) for row in batch)
    if [int(row[0]) for row in rows] != expected:
        raise RuntimeError(f'staged membership mismatch: {path}')
    if any(row[1] != 'pending' for row in rows):
        raise RuntimeError(f'staged status mismatch: {path}')


def _unique_backup(path):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    candidate = f'{path}.previous-{stamp}'
    n = 1
    while os.path.exists(candidate):
        candidate = f'{path}.previous-{stamp}-{n}'
        n += 1
    return candidate


def _install_files(staged_files, manifest_stage, targets, manifest):
    """Install staged files, retaining every replaced old snapshot for rollback."""
    backups = []
    installed = []
    try:
        for target in [*targets, manifest]:
            if os.path.exists(target):
                backup = _unique_backup(target)
                os.replace(target, backup)
                backups.append((target, backup))
        for staged, target in [*zip(staged_files, targets), (manifest_stage, manifest)]:
            os.replace(staged, target)
            installed.append(target)
        return backups, installed
    except Exception:
        _restore_files(backups, installed)
        raise


def _restore_files(backups, installed):
    for target in installed:
        if os.path.exists(target):
            failed = _unique_backup(target)
            os.replace(target, failed)
    for target, backup in reversed(backups):
        if os.path.exists(backup):
            os.replace(backup, target)


def _register_plans(src, plans):
    """Register all plans in one transaction after file staging succeeds."""
    _ensure_canonical(src)
    conn = sqlite3.connect(src, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('BEGIN IMMEDIATE')
        for batch_id, batch in plans:
            pmids = [int(row['pmid']) for row in batch]
            existing = conn.execute(
                'SELECT source, query_text, note FROM batches WHERE batch_id=?',
                (batch_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    'INSERT INTO batches(batch_id, source, query_text, created_at, note) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (batch_id, 'pmc', None, now_str(), BATCH_NOTE),
                )
            elif (existing['source'], existing['query_text'], existing['note']) != (
                'pmc', None, BATCH_NOTE
            ):
                raise ValueError(f'canonical provenance mismatch: {batch_id}')
            for row in batch:
                task = conn.execute(
                    'SELECT status FROM tasks WHERE pmid=?', (row['pmid'],)
                ).fetchone()
                if task is None:
                    raise ValueError(f'task PMID {row["pmid"]} does not exist')
                conn.execute(
                    'INSERT OR IGNORE INTO batch_members(batch_id, pmid, original_status, added_at) '
                    'VALUES (?, ?, ?, ?)',
                    (batch_id, row['pmid'], row['status'], now_str()),
                )
        conn.execute('COMMIT')
    except Exception:
        if conn.in_transaction:
            conn.execute('ROLLBACK')
        raise
    finally:
        conn.close()


def build_batches(src, output_root, *, batch_size=400, max_batches=0, first_batch=1):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if max_batches < 0:
        raise ValueError('max_batches must be non-negative')
    if first_batch <= 0:
        raise ValueError('first_batch must be positive')

    pool = _load_pool(src)
    from collections import Counter
    print(f'候选池: {len(pool)} 篇（有 PMC 且未完成）')
    print('状态分布:', dict(Counter(row['status'] for row in pool)))
    plans = _plans(pool, batch_size, max_batches, first_batch)
    print(f'批次: {len(plans)} 个（每批 {batch_size}）')
    _validate_existing_batches(src, plans)

    reports = os.path.join(output_root, 'reports')
    os.makedirs(reports, exist_ok=True)
    manifest = os.path.join(reports, 'pmc_batches.csv')
    targets = [os.path.join(output_root, f'tasks_pmc_b{i}.sqlite') for i, _ in enumerate(plans, first_batch)]
    with tempfile.TemporaryDirectory(prefix='.pmc-batches-', dir=output_root) as staging:
        staged_files = []
        for index, (_, batch) in enumerate(plans, first_batch):
            staged = os.path.join(staging, f'tasks_pmc_b{index}.sqlite')
            _stage_batch_db(staged, batch)
            _validate_staged_db(staged, batch)
            staged_files.append(staged)
        manifest_stage = os.path.join(staging, 'pmc_batches.csv')
        _stage_manifest(manifest_stage, plans)
        backups, installed = _install_files(staged_files, manifest_stage, targets, manifest)
        try:
            _register_plans(src, plans)
        except Exception:
            _restore_files(backups, installed)
            raise

    for i, (_, batch) in enumerate(plans, first_batch):
        years = [row['year'] for row in batch]
        print(f'b{i}: {targets[i - first_batch]} | {len(batch)} 篇 | 年份 {max(years)}~{min(years)}')
    print(f'清单: {manifest}')
    return {'batches': len(plans), 'members': len(pool), 'manifest': manifest}


def main():
    ap = argparse.ArgumentParser(description='构建 PMC 优先批次队列')
    ap.add_argument('--src', default=os.path.join(ROOT, 'tasks.sqlite'))
    ap.add_argument('--batch-size', type=int, default=400)
    ap.add_argument('--max-batches', type=int, default=0, help='最多生成几批（0=全部）')
    ap.add_argument('--first-batch', type=int, default=1,
                    help='批次起始编号（续跑时用，如 3 表示从 b3 开始命名）')
    args = ap.parse_args()
    build_batches(
        args.src,
        ROOT,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        first_batch=args.first_batch,
    )


if __name__ == '__main__':
    main()
