import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope='module')
def module():
    path = Path(__file__).resolve().parents[3] / 'docs/earth-3hourly-fullgrid-model-template.py'
    spec = importlib.util.spec_from_file_location('full_grid_simvp_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield module
    torch.set_num_threads(previous)


def test_full_grid_simvp_trains_without_spatial_cropping_and_reloads(module):
    config = {'in_channels': 5, 'window': 2, 'horizon': 1, 'spatial_hidden_dim': 4,
              'temporal_hidden_dim': 8, 'num_temporal_blocks': 1, 'dropout': 0.0}
    model = module.build_model(config)
    inputs = torch.randn(1, 2, 5, 240, 480)
    output = model(inputs)
    assert output.shape == (1, 1, 1, 240, 480) and output.dtype == torch.float32
    output.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    restored = module.build_model(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
    with pytest.raises(ValueError, match='240, 480'):
        model(torch.zeros(1, 2, 5, 24, 48))


def test_longitude_convolution_sees_the_opposite_meridian(module):
    convolution = module.LongitudeConv2d(1, 1, kernel_size=3, padding=1, bias=False)
    convolution.weight.data.fill_(1)
    inputs = torch.zeros(1, 1, 5, 10)
    inputs[0,0,2,9] = 1
    output = convolution(inputs)
    assert output[0,0,2,0].item() == 1
    assert output[0,0,2,1].item() == 0
    inputs.zero_()
    inputs[0,0,4,5] = 1
    assert convolution(inputs)[0,0,0,5].item() == 0
