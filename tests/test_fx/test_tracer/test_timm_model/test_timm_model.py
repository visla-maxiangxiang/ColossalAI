import pytest
import timm.models as tm
import torch

from colossalai.fx import symbolic_trace
from tests.kit.model_zoo import model_zoo


def trace_and_compare(model_cls, data, output_transform_fn, meta_args=None):
    # trace
    model = model_cls()

    # convert to eval for inference
    # it is important to set it to eval mode before tracing
    # without this statement, the torch.nn.functional.batch_norm will always be in training mode
    model.eval()

    # TODO: support the following models
    # 1. ConViT
    # 2. NormFreeNet
    # as they are not supported, let's skip them
    if model.__class__.__name__ in ['ConViT', 'NormFreeNet']:
        return

    gm = symbolic_trace(model, meta_args=meta_args)

    # run forward
    with torch.no_grad():
        fx_out = gm(**data)
        non_fx_out = model(**data)

    # compare output
    transformed_fx_out = output_transform_fn(fx_out)
    transformed_non_fx_out = output_transform_fn(non_fx_out)

    assert len(transformed_fx_out) == len(transformed_non_fx_out)

    for key in transformed_fx_out.keys():
        fx_output_val = transformed_fx_out[key]
        non_fx_output_val = transformed_non_fx_out[key]
        assert torch.allclose(fx_output_val, non_fx_output_val, atol=1e-5), \
            f'{model.__class__.__name__} has inconsistent outputs, {fx_output_val} vs {non_fx_output_val}'


def test_timm_models():
    torch.backends.cudnn.deterministic = True

    sub_model_zoo = model_zoo.get_sub_registry('timm')

    for name, (model_fn, data_gen_fn, output_transform_fn, attribute) in sub_model_zoo.items():
        data = data_gen_fn()
        if attribute is not None and attribute.has_control_flow:
            meta_args = {k: v.to('meta') for k, v in data.items()}
        else:
            meta_args = None

        trace_and_compare(model_fn, data, output_transform_fn, meta_args)


if __name__ == '__main__':
    test_timm_models()
