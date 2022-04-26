import pytest
import torch
from detectors.encode import decode, encode, to_cchw, to_coords


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def anchors(batch_size, x0, y0, x1, y1, shift):
    x = torch.zeros((batch_size, 4))
    x[:, 0] = x0 + shift
    x[:, 1] = y0 + shift
    x[:, 2] = x1 + shift
    x[:, 3] = y1 + shift
    return x


@pytest.fixture
def original(batch_size, x0, y0, x1, y1):
    x = torch.zeros((batch_size, 4))
    x[:, 0] = x0
    x[:, 1] = y0
    x[:, 2] = x1
    x[:, 3] = y1
    return x


@pytest.fixture
def encoded(encoder, original, anchors):
    return encoder(original, anchors)


@pytest.fixture
def decoded(decoder, encoded, anchors):
    return decoder(encoded[None], anchors)[0]


@pytest.mark.parametrize("x0, y0, x1, y1", [
    (0.2, 0.2, 0.8, 0.8),
    # (0.1, 0.2, 0.8, 0.8),
    # (0.2, 0.1, 0.8, 0.8),
    # (0.2, 0.1, 0.8, 0.9),
    # (0.1, 0.2, 0.9, 0.8),
])
@pytest.mark.parametrize("shift", [
    -0.1,
    # +0.1,
    # 0.05,
    # +0.05,
    # 0.00,
])
@pytest.mark.parametrize("encoder, decoder", [
    (encode, decode),
    # (lambda x, _: to_coords(x), lambda x, _: to_cchw(x)),
])
def test_encoded_decode_correct(decoded, original):
    torch.testing.assert_close(decoded, original)
