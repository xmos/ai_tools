# Copyright (c) 2020, XMOS Ltd, All rights reserved

import yaml
import logging
import portalocker
import pytest
import _pytest
import numpy as np
from pathlib import Path
from typing import Dict, Type, Optional, Union

from tflite2xcore.utils import dequantize  # type: ignore # TODO: fix this
from tflite2xcore.xcore_model import XCOREModel  # type: ignore # TODO: fix this
from tflite2xcore.model_generation.utils import stringify_config

from . import (
    IntegrationTestRunner,
    DefaultIntegrationTestRunner,
    _compare_batched_arrays,
    BatchedArrayComparison,
)


#  ----------------------------------------------------------------------------
#                                   HOOKS
#  ----------------------------------------------------------------------------


def pytest_addoption(parser):  # type: ignore
    parser.addoption(
        "-C",
        "--coverage",
        action="store",
        default="default",
        choices=["reduced", "default", "extended"],
        help="Set the coverage level",
    )

    parser.addoption(
        "-D",
        "--dump",
        action="store",
        default=None,
        choices=[None, "models"],
        help="Set what contents of the model generation runs should be dumped into cache for easier access.",
    )

    parser.addoption(
        "--config-only",
        action="store_true",
        help="The model generators are configured but not run",
    )

    parser.addoption(
        "--generate-only",
        action="store_true",
        help="The model generators are run and cached but outputs are not evaluated for correctness",
    )

    parser.addoption(
        "--use-device",
        action="store_true",
        help="Execute interpreter on hardware device",
    )


def pytest_generate_tests(metafunc: _pytest.python.Metafunc) -> None:
    if "run" in metafunc.fixturenames:
        try:
            CONFIGS = metafunc.module.CONFIGS
            config_file = Path(metafunc.module.__file__)
        except AttributeError:
            logging.debug(f"CONFIGS undefined in {metafunc.module}")
            config_file = Path(metafunc.module.__file__).with_suffix(".yml")
            try:
                with open(config_file, "r") as f:
                    CONFIGS = yaml.load(f, Loader=yaml.FullLoader)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    "Cannot find .yml test config file and "
                    "test module does not contain CONFIGS"
                ) from e

        coverage = metafunc.config.getoption("coverage")
        try:
            configs = list(CONFIGS[coverage].values())
        except KeyError:
            raise KeyError(
                "CONFIGS does not define coverage level "
                f"'{coverage}' in {config_file.resolve()}"
            ) from None

        metafunc.parametrize(
            "run",
            configs,
            indirect=True,
            ids=[f"CONFIGS[{j}]" for j, _ in enumerate(configs)],
        )


#  ----------------------------------------------------------------------------
#                                   FIXTURES
#  ----------------------------------------------------------------------------


@pytest.fixture(autouse=True)  # type: ignore
def disable_gpus(monkeypatch: _pytest.monkeypatch.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")


_WORKER_CACHE: Dict[Path, IntegrationTestRunner] = {}


@pytest.fixture  # type: ignore
def run(request: _pytest.fixtures.SubRequest) -> IntegrationTestRunner:
    try:
        GENERATOR = request.module.GENERATOR
    except AttributeError:
        raise NameError("GENERATOR not designated in test") from None

    try:
        RUNNER: Type[IntegrationTestRunner] = request.module.RUNNER
    except AttributeError:
        RUNNER = DefaultIntegrationTestRunner

    pytest_config = request.config

    use_device = pytest_config.getoption("--use-device")
    if request.param.pop("skip_on_device", False) and use_device:
        pytest.skip()

    runner = RUNNER(GENERATOR, use_device=use_device)
    runner.set_config(**request.param)

    logging.info(f"Config: {runner._config}")
    if pytest_config.getoption("--config-only"):
        pytest.skip()

    config_str = stringify_config(runner._config)
    file_path = Path(request.module.__file__)
    key = file_path.relative_to(pytest_config.rootdir) / config_str

    try:
        runner = _WORKER_CACHE[key]
    except KeyError:
        dirpath = pytest_config.cache.get(key, "")
        if dirpath:
            runner = runner.load(dirpath)
            logging.debug(f"cached runner loaded from {dirpath}")
            runner.rerun_post_cache()
        else:
            runner.run()
            try:
                with portalocker.BoundedSemaphore(1, hash(key), timeout=0):
                    dirpath = str(pytest_config.cache.makedir("model_cache") / key)
                    dirpath = runner.save(dirpath)
                    if pytest_config.getoption("dump") == "models":
                        runner.dump_models(dirpath)

                    logging.debug(f"runner cached to {dirpath}")
                    pytest_config.cache.set(key, str(dirpath))
            except portalocker.AlreadyLocked:
                # another process will write to cache
                pass
        _WORKER_CACHE[key] = runner

    if pytest_config.getoption("--generate-only"):
        pytest.skip()

    return runner


@pytest.fixture  # type: ignore
def xcore_model(run: IntegrationTestRunner) -> XCOREModel:
    return XCOREModel.deserialize(run._xcore_converter._model)


@pytest.fixture  # type: ignore
def reference_model(run: DefaultIntegrationTestRunner) -> XCOREModel:
    return XCOREModel.deserialize(run.get_xcore_reference_model())


@pytest.fixture  # type: ignore
def abs_output_tolerance() -> int:
    return 1


@pytest.fixture  # type: ignore
def bitpacked_outputs() -> bool:
    return False


@pytest.fixture  # type: ignore
def compared_outputs(
    run: DefaultIntegrationTestRunner,
    abs_output_tolerance: Optional[Union[int, float]],
    bitpacked_outputs: bool,
) -> BatchedArrayComparison:
    if bitpacked_outputs:
        return _compare_batched_arrays(
            run.outputs.xcore, run.outputs.reference_quant, tolerance=0, per_bits=True
        )
    if abs_output_tolerance is None:
        # use implicitly derived tolerance
        output_quantization = run._xcore_evaluator.output_quant
        y_quant = run.outputs.reference_quant
        y_float = run.outputs.reference_float

        # The implicit tolerance is derived from how much the quantized reference
        # deviates from the floating point reference.
        max_diff = np.max(np.abs(dequantize(y_quant, *output_quantization) - y_float))
        # max_diff is usually at least 1 bit, but we ensure this and add some room for error
        abs_output_tolerance = max(float(max_diff), output_quantization.scale) * 1.05
        logging.info(
            f"Using implicit absolute output tolerance: {abs_output_tolerance}"
        )

        return _compare_batched_arrays(
            dequantize(run.outputs.xcore, *output_quantization),
            run.outputs.reference_float,
            abs_output_tolerance,
        )

    return _compare_batched_arrays(
        run.outputs.xcore, run.outputs.reference_quant, abs_output_tolerance
    )
