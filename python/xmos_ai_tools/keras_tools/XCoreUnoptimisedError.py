# Exception to raise if Strictness set to ERROR
class XCoreUnoptimisedError(Exception):
    __layer_idx__: int

    def __init__(self, message: str, layer_idx: int):
        self.__layer_idx__ = layer_idx
        super().__init__(message)

    def get_layer_idx(self) -> int:
        return self.__layer_idx__