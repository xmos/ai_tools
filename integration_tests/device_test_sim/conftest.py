import csv
import shutil
from pathlib import Path

import pytest


REPORT_DIR = Path(__file__).resolve().parent
FIELDS = (
    "model",
    "hw_target",
    "input_mode",
    "status",
    "max_abs_diff",
    "mean_abs_diff",
    "values_outside_tolerance",
    "output_elements",
    "rtol",
    "atol",
    "error",
)
results = {}


def pytest_addoption(parser):
    parser.addoption(
        "--hw-target",
        default="XK-EVK-XU316",
        help=(
            "Simulator target, or REF-XS3/REF-VX4 for Linux TensorFlow Lite; "
            "REF-VX4-CLANG and XK-EVK-XU416-CLANG use VX4 behavior with distinct reports"
        ),
    )


def pytest_sessionstart(session):
    results.clear()
    if hasattr(session.config, "workerinput"):
        return
    build_dir = REPORT_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(call):
    report = yield
    if call.excinfo:
        output = getattr(call.excinfo.value, "stdout", None)
        if output:
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            report.user_properties.append(("error", output.strip()))
    return report


def pytest_runtest_logreport(report):
    properties = dict(report.user_properties)
    row = results.setdefault(report.nodeid, {})
    row.update(properties)
    row.setdefault("model", report.nodeid)
    row.setdefault("status", report.outcome)
    if report.when == "call" or report.failed:
        row["status"] = report.outcome
    if report.failed:
        row["error"] = properties.get("error") or str(report.longrepr).splitlines()[-1]


def pytest_sessionfinish(session, exitstatus):
    # Only the main pytest process writes the combined report.
    if hasattr(session.config, "workerinput"):
        return
    hw_target = session.config.getoption("hw_target")
    report_path = REPORT_DIR / f"model_errors_{hw_target}.csv"
    with report_path.open("w", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results.values(), key=lambda row: (row["model"], row.get("input_mode", ""))))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    terminalreporter.write_sep("-", "Model differences")
    for row in sorted(results.values(), key=lambda row: (row["model"], row.get("input_mode", ""))):
        terminalreporter.write_line(
            f"{row['status'].upper()} {row['model']} [{row.get('input_mode', '')}] "
            f"max_diff={row.get('max_abs_diff', '-')} outside_tol={row.get('values_outside_tolerance', '-')}"
        )
    terminalreporter.write_line(f"Model difference report: {REPORT_DIR / ('model_errors_' + config.getoption('hw_target') + '.csv')}")
    terminalreporter.write_line(f"Work directories (until next run): {REPORT_DIR / 'build'}")
