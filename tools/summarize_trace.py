"""Summarize CUDA activity in an Nsight Systems SQLite export (tested: 2024.5)."""
import argparse
import json
from pathlib import Path
import sqlite3


def summarize(path):
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:
        ranges = db.execute('''SELECT e.start, e.end, COALESCE(e.text, s.value)
            FROM NVTX_EVENTS e LEFT JOIN StringIds s ON s.id=e.textId
            WHERE COALESCE(e.text, s.value) LIKE 'pascal/%' AND e.end IS NOT NULL''').fetchall()
        if not ranges:
            raise ValueError('No pascal/* NVTX ranges. Profile tools/profile_workload.py with --trace=cuda,nvtx.')
        report = []
        for start, end, name in ranges:
            rows = db.execute('''SELECT s.value, COUNT(*), SUM(k.end-k.start), MAX(k.registersPerThread)
                FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
                WHERE k.start>=? AND k.end<=? GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC''', (start, end)).fetchall()
            total = sum(row[2] for row in rows)
            if not total:
                raise ValueError(f'No CUDA kernels in {name}; check process-tree tracing.')
            report.append(dict(name=name, wall_ms=(end-start)/1e6, summed_kernel_ms=total/1e6,
                kernels=[dict(name=n, calls=count, gpu_ms=ns/1e6, share_percent=100*ns/total,
                              max_registers=registers) for n, count, ns, registers in rows]))
        return dict(phases=report, note='Activity durations, not unprofiled throughput; overlapping kernels may sum beyond wall time.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.sqlite), indent=2) + '\n')
