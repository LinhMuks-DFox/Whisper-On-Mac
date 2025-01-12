import whisper
import torch

def get_converted_model(model_name, device="mps"):
    model = whisper.load_model(model_name)
    model = convert_sparse_to_dense(model)
    return model.to(device)

def convert_sparse_to_dense(module):
    for name, param in module.named_parameters():
        if param.is_sparse:
            setattr(module, name, param.to_dense())
    for name, buffer in module.named_buffers():
        if buffer.is_sparse:
            setattr(module, name, buffer.to_dense())
    return module