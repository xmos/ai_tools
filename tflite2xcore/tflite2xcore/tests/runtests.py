#!/usr/bin/env python

import os
import pytest
import argparse
import sys

import multiprocessing as mp

from io import StringIO
from collections import Counter
from timeit import default_timer as timer


class CollectorPlugin:
    def __init__(self, *, mode="files"):
        self.counter = Counter()
        if mode in ["files", "tests"]:
            self.mode = mode
        else:
            raise ValueError(f"Invalid collection mode: '{mode}'")

    def tests(self):
        return self.counter.most_common()

    def pytest_collection_modifyitems(self, items):
        if self.mode is "files":
            self.counter = Counter(item.nodeid.split("::")[0] for item in items)
        elif self.mode is "tests":
            self.counter = Counter(item.nodeid.split("[")[0] for item in items)


class JobCollector:
    def __init__(self, path, *, coverage_options=None, verbose=False, junit=False):
        if not (os.path.exists(path) and os.path.isdir(path)):
            raise ValueError(f"Invalid directory path: {path}")

        self.plugin = CollectorPlugin()
        self.verbose = verbose
        self.jobs = []
        self.path = path
        self.junit = junit

        coverage_options = coverage_options or []
        self.optional_args = ["-qq"] + coverage_options
        self.collection_job = [self.path, "--collect-only"] + self.optional_args

    def collect(self):
        print("Collecting test cases...")
        start = timer()
        sys.stdout = StringIO()
        try:
            exit_code = pytest.main(self.collection_job, plugins=[self.plugin])
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = sys.__stdout__
            print(f"Collection finished in {timer() - start:.2f}s.")

            if self.verbose or exit_code:
                print(output)

        if not exit_code:
            tests = self.plugin.tests()
            for path, _ in tests:
                cmd = [os.path.join(self.path, path), "--tb=short"] + self.optional_args
                if self.junit:
                    cmd += ["--junitxml", path + "_junit.xml"]
                self.jobs.append(cmd)

            print(f"{sum(cnt for _, cnt in tests)} CASES IN {len(self.jobs)} JOBS:")
            for job, (_, cnt) in zip(self.jobs, tests):
                print(f"{cnt} CASES IN: {' '.join(job)}")

        return exit_code


class JobExecutor:
    def __init__(self, job_fun, *, workers=1, verbose=False):
        cpu_count = mp.cpu_count()
        if workers == -1 or workers > cpu_count:
            workers = cpu_count
        elif workers < -1 or workers == 0:
            raise ValueError(f"Invalid number of workers: {workers}")

        self.workers = workers
        self.verbose = verbose
        self.pool = mp.Pool(self.workers)
        self.job_fun = job_fun

    def execute(self, jobs):
        print(f"Executing {len(jobs)} jobs on {self.workers} workers...")

        start = timer()
        outputs = self.pool.map(self.job_fun, jobs)
        total = timer() - start

        passed = failed = 0
        for job, output, t, exit_code in outputs:
            job_str = f"TIME={t:.2f}s in {' '.join(job)}"
            if exit_code:
                failed += 1
                print("FAILED:", job_str)
                if self.verbose:
                    print(output)
                    print()
            else:
                passed += 1
                print("PASSED:", job_str)

        print(f"TOTAL: PASSED={passed}, FAILED={failed}, TIME={total:.2f}s")
        return outputs


def run_job(job):
    sys.stdout = StringIO()
    try:
        import pytest as pt

        start = timer()
        exit_code = pt.main(job)
        t = timer() - start
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__
    return job, output, t, exit_code


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?", default=os.path.curdir)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--extended", action="store_true", default=False)
    parser.add_argument("--collect-only", action="store_true", default=False)
    parser.add_argument("--junit", action="store_true", default=False)
    parser.add_argument("-n", "--workers", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args(raw_args)

    coverage_options = []
    if args.smoke and args.extended:
        raise ValueError('Only one of "--smoke" and "--extended" can be used')
    elif args.smoke:
        coverage_options.append("--smoke")
    elif args.extended:
        coverage_options.append("--extended")

    collector = JobCollector(
        args.dir, coverage_options=coverage_options, verbose=args.verbose, junit=args.junit,
    )
    exit_code = collector.collect()
    if exit_code or args.collect_only or not args.workers:
        exit(exit_code)

    executor = JobExecutor(run_job, workers=args.workers, verbose=args.verbose)
    executor.execute(collector.jobs)


if __name__ == "__main__":
    main()
