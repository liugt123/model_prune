import os
from ctypes import c_char_p, cdll
import tensorrt as trt
import torch
from onnx.backend.base import BackendRep, namedtupledict

import trt.binding as binding

TRT_LOGGER = trt.Logger()

# HACK Should look for a better way/place to do this

libcudart = cdll.LoadLibrary("libcudart.so")
libcudart.cudaGetErrorString.restype = c_char_p


def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + error_string)


class TRT_Engine(BackendRep):
    def __init__(self, engine_file_path, device_id=0, max_batch_size=8):
        self.device_id = device_id
        cudaSetDevice(self.device_id)
        self.max_batch_size = max_batch_size
        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        self._setup_binding()
        self._cuda_stream = torch.cuda.Stream()
        self._context = self.engine.create_execution_context()
        self.dynamic_batch = self.engine.get_binding_shape(0)[0] == -1

    @property
    def cuda_stream(self):
        return self._cuda_stream

    @property
    def device(self) -> str:
        return f"cuda:{self.device_id}"

    def _setup_binding(self):
        self.bindings = [
            binding.Binding(self.engine, i, self.max_batch_size, self.device)
            for i in range(self.engine.num_bindings)
        ]
        self.inputs, self.outputs = [], []
        for b in self.bindings:
            if b.is_input:
                self.inputs.append(b)
            else:
                self.outputs.append(b)

    def run(self, *inputs):
        if len(inputs) != len(self.inputs):
            raise ValueError(
                f"Wrong number of inputs. Expected {len(self.inputs)}, got {len(inputs)}."
            )

        batch_size = inputs[0].size(0)
        if self.dynamic_batch:
            self._context.set_binding_shape(0, inputs[0].shape)
        for _, (input_array, input_binding) in enumerate(zip(inputs, self.inputs)):
            if input_array.shape[0] != batch_size:
                raise ValueError(
                    f"All inputs must have same batch size.Expected {batch_size},\
                    got {input_array.shape[0]}."
                )
            input_binding.data_ptr = input_array.to(self.device)

        binding_addrs = [b.data_ptr for b in self.bindings]
        torch.cuda.synchronize(torch.device(self.device))
        self._context.execute_async_v2(binding_addrs, self.cuda_stream.cuda_stream)
        self.cuda_stream.synchronize()

        output_names = [output.name for output in self.outputs]
        outputs = [output.data(batch_size) for output in self.outputs]
        outputs_tuple = namedtupledict("Outputs", output_names)(*outputs)

        return outputs_tuple

    __call__ = run
