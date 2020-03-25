#!/usr/bin/env python

import pytest
import argparse
import sys
import pathlib
from timeit import default_timer as timer

import multiprocessing as mp

from io import StringIO


def run_job(job):
    sys.stdout = StringIO()
    t = None
    try:
        import pytest as pt
        start = timer()
        exit_code = pt.main(job)
        t = timer() - start
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__
    return job, output, t, exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', default=False)
    parser.add_argument('--collect-only', action='store_true', default=False)
    parser.add_argument('-n', '--workers', type=int, default=1)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()

    cpu_count = mp.cpu_count()
    if args.workers == -1 or args.workers > cpu_count:
        args.workers = cpu_count
    elif args.workers == 0:
        args.collect_only = True
    elif args.workers < -1:
        raise argparse.ArgumentTypeError(f"Invalid number of workers: {args.workers}")

    print("Collecting test cases...")
    start = timer()
    sys.stdout = StringIO()
    try:
        collect_job = [".", "--collect-only", "-qq"]
        if args.smoke:
            collect_job.append("--smoke")
        exit_code = pytest.main(collect_job)
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__
        print(f"Collection finished in {timer() - start:.2f}s.")

        if args.verbose or exit_code:
            print(output)
        if exit_code:
            exit(exit_code)

    def line_parser(line):
        parts = line.split(' ')
        if len(parts) != 2:
            return
        path, cnt = parts
        path = pathlib.Path(path[:-1])  # ends with a colon
        if path.suffix != '.py':
            return
        return int(cnt), str(path)

    testfiles = sorted(
        [line_parser(line) for line in output.split('\n')
         if line.startswith("test_")],
        reverse=True
    )
    jobs = [[path, "-qq", "--tb=short"] + (["--smoke"] if args.smoke else [])
            for cnt, path in testfiles]
    print(f"{sum(cnt for cnt, _ in testfiles)} CASES IN {len(jobs)} JOBS:")
    for job, (cnt, _) in zip(jobs, testfiles):
        print(f"{cnt} CASES IN: {' '.join(job)}")

    if args.collect_only:
        exit()

    print()
    print(f"Executing jobs on {args.workers} workers...")
    pool = mp.Pool(args.workers)
    start = timer()
    outputs = pool.map(run_job, jobs)
    total = timer() - start

    passed = failed = 0
    for job, output, t, exit_code in outputs:
        job_str = f"TIME={t:.2f}s in {' '.join(job)}"
        if exit_code:
            failed += 1
            print("FAILED:", job_str)
            if args.verbose:
                print(output)
                print()
        else:
            passed += 1
            print("PASSED:", job_str)

    print(f"TOTAL: PASSED={passed}, FAILED={failed}, TIME={total:.2f}s")
