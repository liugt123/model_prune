import builtins
from typing import Union
import six
import tensorrt as trt
import torch

_TYPE_MAPPING = [
    (torch.float32, trt.DataType.FLOAT),
    (torch.float16, trt.DataType.HALF),
    (torch.int8, trt.DataType.INT8),
]

TYPE_TRT_2_TORCH = {o: t for t, o in _TYPE_MAPPING}
TYPE_TORCH_2_TRT = {t: o for t, o in _TYPE_MAPPING}


class Binding:
    def __init__(
        self,
        engine: trt.ICudaEngine,
        idx_or_name: Union[int, str],
        max_batch_size: int,
        device: str,
    ):
        if isinstance(idx_or_name, six.string_types):
            self.name = idx_or_name
            self.index = engine.get_binding_index(self.name)
            if self.index == -1:
                raise IndexError(f"Binding name not found: {self.name}")
        else:
            self.index = idx_or_name
            self.name = engine.get_binding_name(self.index)
            if self.name is None:
                raise IndexError(f"Binding index out of range: {self.index}")

        self._dtype = TYPE_TRT_2_TORCH[engine.get_binding_dtype(self.index)]
        self._shape = (max_batch_size,) + tuple(engine.get_binding_shape(self.index))[1:]
        self._device = torch.device(device)
        self._is_input = engine.binding_is_input(self.index)
        if self.is_input:
            self._binding_data = None
        else:
            self._binding_data = torch.zeros(size=self.shape, dtype=self.dtype, device=self.device)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return self._device

    @property
    def is_input(self) -> bool:
        return self._is_input

    def data(self, batch_size):
        return self._binding_data[:batch_size, ...].clone()

    @property
    def data_ptr(self) -> builtins.int:
        if self._binding_data is None:
            raise ValueError("miss bind input data.")
        return self._binding_data.data_ptr()

    @data_ptr.setter
    def data_ptr(self, value: torch.Tensor):
        assert self.is_input, f"only input data could be bind, self.is_input={self.is_input}."
        self._binding_data = value
