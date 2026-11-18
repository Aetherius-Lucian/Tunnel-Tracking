import torch

from msfw_block import MSFWBlock


def _run_case(shape: tuple[int, int, int, int]) -> None:
    x = torch.randn(*shape, requires_grad=True)
    block = MSFWBlock(shape[1])
    y = block(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None


def test_msfw_block_shapes_and_backward() -> None:
    for shape in ((2, 16, 64, 64), (2, 16, 80, 96), (2, 16, 79, 95)):
        _run_case(shape=shape)
