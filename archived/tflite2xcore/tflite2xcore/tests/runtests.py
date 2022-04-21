#!/usr/bin/env python
# Copyright 2020-2021 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.

import os
import pytest
import argparse
import sys
import atexit

import multiprocessing as mp

from io import StringIO
from enum import Enum, auto
from timeit import default_timer as timer
from typing import Counter, List, Tuple, Optional, NamedTuple, Callable, Sequence


class CollectionMode(Enum):
    FILES = auto()
    TESTS = auto()


class CollectorPlugin:
    def __init__(self, *, mode: CollectionMode = CollectionMode.FILES) -> None:
        self.counter = Counter[str]()
        self.mode = mode

    def tests(self) -> List[Tuple[str, int]]:
        return self.counter.most_common()

    def pytest_collection_modifyitems(self, items: List[pytest.Item]) -> None:
        if self.mode is CollectionMode.FILES:
            self.counter = Counter(item.nodeid.split("::")[0] for item in items)
        elif self.mode is CollectionMode.TESTS:
            self.counter = Counter(item.nodeid.split("[")[0] for item in items)
        else:
            raise ValueError(f"Unsupported collection mode {self.mode}")


Job = List[str]


class JobCollector:
    def __init__(
        self,
        path: str,
        *,
        coverage_options: Optional[Sequence[str]] = None,
        verbose: bool = False,
        junit: bool = False,
    ) -> None:
        if not (os.path.exists(path) and os.path.isdir(path)):
            raise ValueError(f"Invalid directory path: {path}")

        self.plugin = CollectorPlugin()
        self.verbose = verbose
        self.jobs: List[Job] = []
        self.path = path
        self.junit = junit

        coverage_options = list(coverage_options or [])
        self.optional_args = ["-qq"] + coverage_options
        self.collection_job = [self.path, "--collect-only"] + self.optional_args

    def collect(self) -> int:
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
            self.jobs = []
            tests = self.plugin.tests()
            for path, _ in tests:
                full_path = os.path.join(self.path, path)
                cmd = [full_path, "--tb=short"] + self.optional_args
                if self.junit:
                    cmd += ["--junitxml", full_path + "_junit.xml"]
                self.jobs.append(cmd)

            print(f"{sum(cnt for _, cnt in tests)} CASES IN {len(self.jobs)} JOBS:")
            for job, (_, cnt) in zip(self.jobs, tests):
                print(f"{cnt} CASES IN: {' '.join(job)}")

        return exit_code


class JobResult(NamedTuple):
    job: Job
    output: str
    time: float
    exit_code: int


class JobExecutor:
    def __init__(
        self,
        job_fun: Callable[[Job], JobResult],
        *,
        workers: int = 1,
        verbose: bool = False,
    ) -> None:
        cpu_count = mp.cpu_count()
        if workers == -1 or workers > cpu_count:
            workers = cpu_count
        elif workers < -1 or workers == 0:
            raise ValueError(f"Invalid number of workers: {workers}")

        self.workers = workers
        self.verbose = verbose
        self.pool = mp.Pool(self.workers)
        atexit.register(self.pool.close)
        self.job_fun = job_fun

    def execute(self, jobs: Sequence[Job]) -> Sequence[JobResult]:
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


def run_job(job: Job) -> JobResult:
    sys.stdout = StringIO()
    try:
        start = timer()
        exit_code = pytest.main(job)
        t = timer() - start
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__
    return JobResult(job, output, t, exit_code)


def main(raw_args: Optional[Sequence[str]] = None) -> None:
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
        args.dir,
        coverage_options=coverage_options,
        verbose=args.verbose,
        junit=args.junit,
    )
    exit_code = collector.collect()
    if exit_code or args.collect_only or not args.workers:
        exit(exit_code)

    executor = JobExecutor(run_job, workers=args.workers, verbose=args.verbose)
    executor.execute(collector.jobs)


if __name__ == "__main__":
    main()
