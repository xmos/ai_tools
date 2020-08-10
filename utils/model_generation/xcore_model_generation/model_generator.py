# Copyright (c) 2020, XMOS Ltd, All rights reserved

import dill  # type: ignore
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
import tensorflow as tf  # type: ignore

from tflite2xcore.utils import set_all_seeds  # type: ignore # TODO: fix this
from tflite2xcore import xlogging  # type: ignore # TODO: fix this
from tflite2xcore import tflite_visualize


from .model_converter import (
    ModelConverter,
    TFLiteQuantConverter,
    XCoreConverter,
)
from .model_runner import ModelRunner, IntegrationTestRunner
from .model_evaluator import ModelEvaluator, TFLiteQuantEvaluator, XCoreEvaluator

from typing import Optional, Tuple, List, Dict, Any, Union


Configuration = Dict[str, Any]


class ModelGenerator(ABC):
    """ Superclass for defining parameterized automatic model generation.

    The main use case of this class is generation of the integration test
    models.
    """

    _model: Any
    _config: Configuration = {}
    run: ModelRunner

    def __init__(
        self,
        runner: ModelRunner,
        converters: Optional[List[ModelConverter]] = None,
        evaluators: Optional[List[ModelEvaluator]] = None,
    ) -> None:
        """ Registers the runner, converters and evaluators. """
        self.run = runner
        self._converters = converters or []
        self._evaluators = evaluators or []

    @abstractmethod
    def build(self) -> None:
        """ Sets the _model field as needed by the subclass.
        
        The configuration should be set using the set_config method before
        calling this.
        """
        raise NotImplementedError()

    @abstractmethod
    def _set_config(self, cfg: Configuration) -> None:
        """ Sets the relevant configuration parameters and returns the unused ones.
        
        Should check if the given configuration parameters are legal.
        This method operates on the config input argument in-place.
        Subclasses should implement this instead of the set_config method.
        """
        for converter in self._converters:
            converter._set_config(cfg)

    def set_config(self, **config: Any) -> None:
        """ Configures the model generator before the build method is run.
        
        Default values for missing configuration parameters are set.
        Subclasses should implement the _set_config method instead of this.
        """
        self._set_config(config)
        if config:
            raise ValueError(
                f"Unexpected configuration parameter(s): {', '.join(config.keys())}"
            )


class KerasModelGenerator(ModelGenerator):
    _model: tf.keras.Model

    def _prep_backend(self) -> None:
        tf.keras.backend.clear_session()
        set_all_seeds()

    @property
    def _input_shape(self) -> Tuple[int, ...]:
        return self._model.input_shape[1:]  # type:ignore  # pylint: disable=no-member

    @property
    def _output_shape(self) -> Tuple[int, ...]:
        return self._model.output_shape[1:]  # type:ignore  # pylint: disable=no-member

    def save(self, dirpath: Union[Path, str]) -> Path:
        """ Saves the model contents to the specified directory.
        
        If the directory doesn't exist, it is created.
        If the directory is not empty, it is purged.
        """
        dirpath = Path(dirpath)
        if dirpath.exists():
            shutil.rmtree(dirpath)
        dirpath.mkdir(parents=True)
        self._model.save(dirpath / "model.h5")
        tmp = self._model
        del self._model
        with open(dirpath / "generator.dill", "wb") as f:
            dill.dump(self, f)
        self._model = tmp
        return dirpath

    @classmethod
    def load(cls, dirpath: Union[Path, str]) -> "KerasModelGenerator":
        dirpath = Path(dirpath)
        with open(dirpath / "generator.dill", "rb") as f:
            obj = dill.load(f)
        assert isinstance(obj, cls)

        # tf may complain about missing training config, so silence it
        with xlogging.LoggingContext(tf.get_logger(), xlogging.ERROR):
            obj._model = tf.keras.models.load_model(dirpath / "model.h5")
        return obj


class IntegrationTestModelGenerator(KerasModelGenerator):
    _reference_converter: TFLiteQuantConverter
    _xcore_converter: XCoreConverter
    reference_evaluator: TFLiteQuantEvaluator
    xcore_evaluator: XCoreEvaluator
    run: IntegrationTestRunner

    def __init__(self) -> None:
        self._reference_converter = TFLiteQuantConverter(self)
        self._xcore_converter = XCoreConverter(self, self._reference_converter)
        self.xcore_evaluator = XCoreEvaluator(
            self._reference_converter._get_representative_data,
            lambda: self._xcore_converter._model,
        )
        self.reference_evaluator = TFLiteQuantEvaluator(
            lambda: self.xcore_evaluator.input_data_float,
            lambda: self._reference_converter._model,
            lambda: self.xcore_evaluator.input_quant,
            lambda: self.xcore_evaluator.output_quant,
        )

        super().__init__(
            runner=IntegrationTestRunner(self),
            converters=[self._reference_converter, self._xcore_converter],
            evaluators=[self.xcore_evaluator, self.reference_evaluator],
        )

    def save(self, dirpath: Union[Path, str], dump_models: bool = False) -> Path:
        dirpath = super().save(dirpath)
        if dump_models:
            for name, model in [
                ("model_ref", self._reference_converter._model),
                ("model_xcore", self._xcore_converter._model),
            ]:
                model_ref_path = (dirpath / name).with_suffix(".tflite")
                model_ref_html = model_ref_path.with_suffix(".html")
                with open(model_ref_path, "wb") as f:
                    f.write(model)
                xlogging.logging.debug(f"{name} dumped to {model_ref_path}")
                tflite_visualize.main(model_ref_path, model_ref_html)
                xlogging.logging.debug(
                    f"{name} visualization dumped to {model_ref_html}"
                )
        return dirpath

