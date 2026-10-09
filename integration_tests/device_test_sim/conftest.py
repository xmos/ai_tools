import csv
from pathlib import Path

import pytest


REPORT_DIR = Path(__file__).resolve().parent
FIELDS = (
    "model", "hw_target", "status", "max_abs_diff", "mean_abs_diff",
    "values_outside_tolerance", "total_values", "rtol", "atol", "error",
)
results = {}


def pytest_addoption(parser):
    parser.addoption(
        "--hw-target", default="XK-EVK-XU316",
        help="Hardware target used to compile simulator models (VX4: XK-EVK-XU416)",
    )


def pytest_sessionstart(session):
    results.clear()


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
    test_file, _, test_name = report.nodeid.partition("::")
    if not test_file.endswith("test_device_sim.py") or not test_name.startswith("test_pipeline["):
        return
    if report.when != "call" and not report.failed:
        return
    properties = dict(report.user_properties)
    row = results.setdefault(report.nodeid, {"model": properties.get("model", test_name)})
    row.update({field: properties[field] for field in FIELDS if field in properties})
    if row.get("status") != "failed":
        row["status"] = report.outcome
    if report.failed:
        row["error"] = properties.get("error") or str(report.longrepr).splitlines()[-1]


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput") or not results:
        return
    hw_target = session.config.getoption("hw_target")
    report_path = REPORT_DIR / f"model_errors_{hw_target}.csv"
    with report_path.open("w", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(results.values(), key=lambda row: row["model"]))
    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal:
        terminal.write_line("")
        terminal.write_sep("-", "Model differences")
        terminal.write_line(report_path.read_text().rstrip())
        terminal.write_line(f"Model difference report: {report_path}")
