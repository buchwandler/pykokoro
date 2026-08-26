from types import SimpleNamespace

import numpy as np
import pytest

from pykokoro.audio_generator import AudioGenerator


class Session:
    def __init__(self, inputs):
        self._inputs = inputs

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return []


@pytest.mark.parametrize(
    ("token_name", "speed_type", "expected_speed"),
    [
        ("tokens", "tensor(float)", np.float32),
        ("input_ids", "tensor(float)", np.float32),
        ("input_ids", "tensor(int32)", np.int32),
    ],
)
def test_onnx_inputs_follow_metadata(token_name, speed_type, expected_speed):
    session = Session(
        [
            SimpleNamespace(name=token_name, type="tensor(int64)"),
            SimpleNamespace(name="style", type="tensor(float)"),
            SimpleNamespace(name="speed", type=speed_type),
        ]
    )
    generator = AudioGenerator(session=session, tokenizer=object(), model_source="github")

    inputs = generator._build_onnx_inputs([[0, 1, 0]], np.zeros((1, 256)), 1.125)

    assert set(inputs) == {token_name, "style", "speed"}
    assert inputs[token_name].dtype == np.int64
    assert inputs["style"].dtype == np.float32
    assert inputs["speed"].dtype == expected_speed
    if np.issubdtype(expected_speed, np.floating):
        assert inputs["speed"].item() == pytest.approx(1.125)
    else:
        assert inputs["speed"].item() == 1


def test_nabra_uses_ref_s_input_name():
    session = Session(
        [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="ref_s", type="tensor(float)"),
            SimpleNamespace(name="speed", type="tensor(float)"),
        ]
    )
    generator = AudioGenerator(session=session, tokenizer=object(), model_source="github")

    inputs = generator._build_onnx_inputs([[0, 1, 0]], np.zeros((1, 256)), 1.0)

    assert set(inputs) == {"input_ids", "ref_s", "speed"}
    assert inputs["input_ids"].dtype == np.int64
    assert inputs["ref_s"].dtype == np.float32
    assert inputs["speed"].dtype == np.float32
