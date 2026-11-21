# -*- coding: utf-8 -*-
"""阶段 11：合并批次结果，同时保留 canonical 与全部历史表。

合并只更新现有 canonical ``tasks`` 的 legacy 任务字段，并以 upsert 方式补充
``run_history``。数据库先复制到临时文件，再校验任务数、历史数和所有既有表，
通过后才原子替换；不会删除批次库、A1 provenance 或 A8 batches 历史。
"""

import csv
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a1_schema import migrate_to_v2
from common import ROOT
from lease_batch import ensure_lease_schema

BATCHES = [
    ('sample_50',   'tasks_sample_50.sqlite'),
    ('sample_1000', 'tasks_sample.sqlite'),
    ('template',    'tasks_template.sqlite'),
    ('batch810',    'tasks_batch810.sqlite'),
    ('test3',       'tasks_test3.sqlite'),
    ('test10',      'tasks_test10.sqlite'),
    ('validate400', 'tasks_validate400.sqlite'),
    ('pmc_b1',      'tasks_pmc_b1.sqlite'),
    ('pmc_b2',      'tasks_pmc_b2.sqlite'),
    ('pmc_b3',      'tasks_pmc_b3.sqlite'),
    ('pmc_b4',      'tasks_pmc_b4.sqlite'),
    ('pmc_b5',      'tasks_pmc_b5.sqlite'),
]
TASKS_COLS = ['pmid', 'doi', 'pmc', 'year', 'title', 'title_norm', 'status',
              'attempts', 'route', 'last_error', 'pdf_path', 'license', 'updated_at']
HISTORY_COLS = ['run'] + TASKS_COLS
ARCHIVE_NOTE = '2000年前，封存不爬取'
MERGED_DIR = os.path.join(ROOT, 'pdfs_merged')


def read_batch(db_path):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cols = [r[1] for r in cur.execute('PRAGMA table_info(tasks)').fetchall()]
        rows = [dict(zip(cols, r)) for r in cur.execute('SELECT * FROM tasks').fetchall()]
        for row in rows:
            row.setdefault('license', 'unverified')
        return rows
    finally:
        conn.close()


def _normalize_history_row(run, row):
    result = {'run': run}
    for col in TASKS_COLS:
        value = row.get(col)
        result[col] = None if value == '' else value
    result['license'] = result.get('license') or 'unverified'
    if result.get('pmid') is not None:
        result['pmid'] = int(result['pmid'])
    if result.get('year') is not None:
        result['year'] = int(result['year'])
    if result.get('attempts') is not None:
        result['attempts'] = int(result['attempts'])
    return result


def load_run_rows():
    """优先读 reports/run_history.csv，缺失时回退读批次库。"""
    hist_csv = os.path.join(ROOT, 'reports', 'run_history.csv')
    if os.path.exists(hist_csv):
        with open(hist_csv, encoding='utf-8-sig', newline='') as f:
            rd = csv.DictReader(f)
            rows = [
                _normalize_history_row(r.get('Run') or r.get('run'), r)
                for r in rd
            ]
        print(f'run_history.csv 读取: {len(rows)} 行')
        return rows
    rows = []
    for run, fname in BATCHES:
        path = os.path.join(ROOT, fname)
        if os.path.exists(path):
            rows.extend(_normalize_history_row(run, row) for row in read_batch(path))
    return rows


def _existing_history(conn):
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='run_history'"
    ).fetchone()
    if table is None:
        return []
    columns = [row[1] for row in conn.execute('PRAGMA table_info(run_history)')]
    rows = []
    for values in conn.execute('SELECT * FROM run_history').fetchall():
        raw = dict(zip(columns, values))
        rows.append(_normalize_history_row(raw.get('run') or raw.get('Run'), raw))
    return rows


def _merge_history(existing, incoming):
    """Retain old rows and replace only the same (run, PMID) snapshot."""
    merged = {}
    order = []
    for row in existing + incoming:
        key = (row.get('run'), row.get('pmid'))
        if key not in merged:
            order.append(key)
        merged[key] = row
    return [merged[key] for key in order]


def _history_table(conn):
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='run_history'"
    ).fetchone()
    if table is None:
        conn.execute('''CREATE TABLE run_history (
            run TEXT, pmid INTEGER, doi TEXT, pmc TEXT, year INTEGER, title TEXT,
            title_norm TEXT, status TEXT, attempts INTEGER, route TEXT,
            last_error TEXT, pdf_path TEXT, license TEXT, updated_at TEXT)''')
    columns = [row[1] for row in conn.execute('PRAGMA table_info(run_history)')]
    missing = [col for col in ('run', 'pmid') if col not in columns]
    if missing:
        raise RuntimeError('run_history missing required columns: ' + ', '.join(missing))
    return columns


def _upsert_history(conn, rows):
    columns = _history_table(conn)
    writable = [col for col in HISTORY_COLS if col in columns]
    for row in rows:
        values = [row.get(col) for col in writable]
        existing = conn.execute(
            'SELECT rowid FROM run_history WHERE run=? AND pmid=? LIMIT 1',
            (row.get('run'), row.get('pmid')),
        ).fetchone()
        if existing is None:
            placeholders = ','.join('?' for _ in writable)
            conn.execute(
                f'INSERT INTO run_history ({",".join(writable)}) VALUES ({placeholders})',
                values,
            )
        else:
            assignments = ','.join(f'{col}=?' for col in writable)
            conn.execute(
                f'UPDATE run_history SET {assignments} WHERE rowid=?',
                [*values, existing[0]],
            )


def _copy_database(source, target):
    if os.path.exists(target):
        os.remove(target)
    src = sqlite3.connect(source)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def _temporary_backup(path):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    candidate = f'{path}.premerge-{stamp}'
    suffix = 1
    while os.path.exists(candidate):
        candidate = f'{path}.premerge-{stamp}-{suffix}'
        suffix += 1
    return candidate


def _write_history_csv(path, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Run'] + TASKS_COLS)
        for row in rows:
            writer.writerow([row.get('run')] + [row.get(col) for col in TASKS_COLS])


def main():
    old = os.path.join(ROOT, 'tasks.sqlite')
    migrate_to_v2(old)
    ensure_lease_schema(old)

    source = sqlite3.connect(old)
    source.row_factory = sqlite3.Row
    try:
        base = [dict(row) for row in source.execute(
            f'SELECT {",".join(TASKS_COLS)} FROM tasks'
        ).fetchall()]
        existing_history = _existing_history(source)
        preserved_counts = {}
        for table in ('batches', 'batch_members', 'fetch_attempts',
                      'source_candidates', 'discovery_batches', 'task_discoveries'):
            exists = source.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            preserved_counts[table] = (
                int(source.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])
                if exists else None
            )
    finally:
        source.close()

    incoming = load_run_rows()
    history = _merge_history(existing_history, incoming)
    per_batch = {}
    for row in history:
        per_batch[row['run']] = per_batch.get(row['run'], 0) + 1
    print('各批次行数:', per_batch, '合计', len(history))
    print('tasks.sqlite 基础行:', len(base))

    by_pmid = {}
    for row in history:
        by_pmid.setdefault(row['pmid'], []).append(row)
    merged_pdf = {}
    if os.path.isdir(MERGED_DIR):
        for fn in os.listdir(MERGED_DIR):
            if fn.startswith('PMID_') and fn.endswith('.pdf'):
                merged_pdf[int(fn[5:-4])] = os.path.join(MERGED_DIR, fn)

    def derive(row):
        pmid = row['pmid']
        if row['year'] < 2000:
            return 'archived', None, None, ARCHIVE_NOTE, 0, None, row.get('license') or 'unverified'
        runs = by_pmid.get(pmid, [])
        if not runs:
            return (row['status'], row['route'], row['pdf_path'], row['last_error'],
                    row['attempts'] or 0, row['updated_at'], row.get('license') or 'unverified')
        attempts = sum((run['attempts'] or 0) for run in runs)
        updated = max((run['updated_at'] or '' for run in runs), default=None)
        for run in runs:
            if run['status'] == 'done':
                pdf = merged_pdf.get(pmid) or run['pdf_path']
                return 'done', run['route'], pdf, None, attempts, updated, run.get('license') or row.get('license') or 'unverified'
        latest = max(runs, key=lambda run: run['updated_at'] or '')
        return (latest['status'], latest['route'], latest['pdf_path'], latest['last_error'],
                attempts, updated, latest.get('license') or row.get('license') or 'unverified')

    derived = []
    for row in base:
        status, route, pdf, error, attempts, updated, license = derive(row)
        row.update({
            'status': status, 'route': route, 'pdf_path': pdf, 'last_error': error,
            'attempts': attempts, 'license': license, 'updated_at': updated,
        })
        derived.append(row)

    tmp = old + '.merge.tmp'
    hist_csv = os.path.join(ROOT, 'reports', 'run_history.csv')
    hist_tmp = hist_csv + '.merge.tmp'
    _copy_database(old, tmp)
    try:
        conn = sqlite3.connect(tmp)
        try:
            conn.execute('BEGIN IMMEDIATE')
            update_cols = [col for col in TASKS_COLS if col != 'pmid']
            assignments = ','.join(f'{col}=?' for col in update_cols)
            for row in derived:
                cursor = conn.execute(
                    f'UPDATE tasks SET {assignments} WHERE pmid=?',
                    [row[col] for col in update_cols] + [row['pmid']],
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f'missing task PMID during merge: {row["pmid"]}')
            _upsert_history(conn, history)
            task_count = int(conn.execute('SELECT COUNT(*) FROM tasks').fetchone()[0])
            history_count = int(conn.execute('SELECT COUNT(*) FROM run_history').fetchone()[0])
            for table, expected in preserved_counts.items():
                if expected is None:
                    continue
                actual = int(conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0])
                if actual != expected:
                    raise RuntimeError(f'{table} count changed during merge: {expected} -> {actual}')
            if task_count != len(base) or history_count < len(existing_history):
                raise RuntimeError(
                    f'merge validation failed: tasks {len(base)} -> {task_count}, '
                    f'run_history {len(existing_history)} -> {history_count}'
                )
            conn.commit()
            print(f'新库写入: tasks={task_count}, run_history={history_count}')
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

        os.makedirs(os.path.dirname(hist_csv), exist_ok=True)
        _write_history_csv(hist_tmp, history)
        backup_db = _temporary_backup(old)
        os.replace(old, backup_db)
        try:
            os.replace(tmp, old)
        except Exception:
            os.replace(backup_db, old)
            raise
        os.replace(hist_tmp, hist_csv)
        print(f'tasks.sqlite 已替换；旧库保留为 {os.path.basename(backup_db)}')
        print(f'导出: {hist_csv}（{len(history)} 行）')
    except Exception:
        if os.path.exists(tmp):
            print('!! 校验未通过，原 tasks.sqlite 与历史保持不变；临时文件保留:', tmp)
        if os.path.exists(hist_tmp):
            print('!! 清单临时文件保留:', hist_tmp)
        raise


if __name__ == '__main__':
    main()
