# Copyright (c) 2022–2025 China Merchants Research Institute of Advanced Technology Corporation and its Affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test various scenarios and edge cases for DataProto.chunk method."""

from l0.verl_adapter import monkey_patch_tensordict_chunk

monkey_patch_tensordict_chunk()
import numpy as np
import torch
from verl.protocol import DataProto


def test_chunk_basic_divisible():
    """Test Case 1: Basic divisible case - 8/4."""
    print("=== Test Case 1: 8/4 (divisible) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(8)}, meta_info={"test": "case1"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert chunks[0].batch["values"].tolist() == [0, 1]
    assert chunks[1].batch["values"].tolist() == [2, 3]
    assert chunks[2].batch["values"].tolist() == [4, 5]
    assert chunks[3].batch["values"].tolist() == [6, 7]
    print("✓ Test Case 1 passed")


def test_chunk_not_divisible():
    """Test Case 2: Non-divisible case - 9/4."""
    print("=== Test Case 2: 9/4 (non-divisible) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(9)}, meta_info={"test": "case2"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert chunks[0].batch["values"].tolist() == [0, 1, 2]
    assert chunks[1].batch["values"].tolist() == [3, 4]
    assert chunks[2].batch["values"].tolist() == [5, 6]
    assert chunks[3].batch["values"].tolist() == [7, 8]
    print("✓ Test Case 2 passed")


def test_chunk_data_less_than_chunks():
    """Test Case 3: Data size less than chunk count - 3/4."""
    print("=== Test Case 3: 3/4 (data size less than chunk count) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(3)}, meta_info={"test": "case3"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert chunks[0].batch["values"].tolist() == [0]
    assert chunks[1].batch["values"].tolist() == [1]
    assert chunks[2].batch["values"].tolist() == [2]
    assert chunks[3].batch["values"].tolist() == []  # auto padding
    print("✓ Test Case 3 passed")


def test_chunk_single_data():
    """Test Case 4: Single data - 1/4."""
    print("=== Test Case 4: 1/4 (single data) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(1)}, meta_info={"test": "case4"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert chunks[0].batch["values"].tolist() == [0]
    assert chunks[1].batch["values"].tolist() == []  # auto padding
    assert chunks[2].batch["values"].tolist() == []  # auto padding
    assert chunks[3].batch["values"].tolist() == []  # auto padding
    print("✓ Test Case 4 passed")


def test_chunk_large_data():
    """Test Case 5: Large data size - 100/4."""
    print("=== Test Case 5: 100/4 (large data size) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(100)}, meta_info={"test": "case5"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert len(chunks[0].batch["values"]) == 25
    assert chunks[0].batch["values"][0].item() == 0
    assert chunks[3].batch["values"][-1].item() == 99
    print("✓ Test Case 5 passed")


def test_chunk_large_data_not_divisible():
    """Test Case 6: Large data size non-divisible - 101/4."""
    print("=== Test Case 6: 101/4 (large data size non-divisible) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(101)}, meta_info={"test": "case6"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert len(chunks[0].batch["values"]) == 26
    assert len(chunks[1].batch["values"]) == 25
    assert len(chunks[2].batch["values"]) == 25
    assert len(chunks[3].batch["values"]) == 25
    assert chunks[0].batch["values"][0].item() == 0
    assert chunks[3].batch["values"][-1].item() == 100
    print("✓ Test Case 6 passed")


def test_chunk_multi_dimensional():
    """Test Case 7: Multi-dimensional tensor - 12/3."""
    print("=== Test Case 7: 12/3 (multi-dimensional tensor) ===")
    data = DataProto.from_dict(tensors={"values": torch.randn(12, 5, 3)}, meta_info={"test": "case7"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(3)
    assert len(chunks) == 3
    assert chunks[0].batch["values"].shape == (4, 5, 3)
    assert chunks[1].batch["values"].shape == (4, 5, 3)
    assert chunks[2].batch["values"].shape == (4, 5, 3)
    print("✓ Test Case 7 passed")


def test_chunk_multiple_tensors():
    """Test Case 8: Multiple tensor fields - 10/2."""
    print("=== Test Case 8: 10/2 (multiple tensor fields) ===")
    data = DataProto.from_dict(
        tensors={"values": torch.arange(10), "features": torch.randn(10, 8), "labels": torch.randint(0, 5, (10,))},
        meta_info={"test": "case8"},
    )
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(2)
    assert len(chunks) == 2
    assert chunks[0].batch["values"].shape == (5,)
    assert chunks[0].batch["features"].shape == (5, 8)
    assert chunks[0].batch["labels"].shape == (5,)
    assert chunks[1].batch["values"].shape == (5,)
    print("✓ Test Case 8 passed")


def test_chunk_with_non_tensor_batch():
    """Test Case 9: With non_tensor_batch - 8/4."""
    print("=== Test Case 9: 8/4 (with non_tensor_batch) ===")
    data = DataProto.from_dict(
        tensors={"values": torch.arange(8)},
        non_tensors={"strings": ["a", "b", "c", "d", "e", "f", "g", "h"]},
        meta_info={"test": "case9"},
    )
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert chunks[0].batch["values"].tolist() == [0, 1]
    assert chunks[0].non_tensor_batch["strings"].tolist() == ["a", "b"]
    assert chunks[1].batch["values"].tolist() == [2, 3]
    assert chunks[1].non_tensor_batch["strings"].tolist() == ["c", "d"]
    print("✓ Test Case 9 passed")


def test_chunk_empty_data():
    """Test Case 10: Edge case - 0/4."""
    print("=== Test Case 10: 0/4 (empty data) ===")
    data = DataProto.from_dict(tensors={"values": torch.empty(0)}, meta_info={"test": "case10"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk.batch["values"]) == 0
    print("✓ Test Case 10 passed")


def test_chunk_large_chunk_count():
    """Test Case 11: Large chunk count - 20/10."""
    print("=== Test Case 11: 20/10 (large chunk count) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(20)}, meta_info={"test": "case11"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(10)
    assert len(chunks) == 10
    for i, chunk in enumerate(chunks):
        assert len(chunk.batch["values"]) == 2
        assert chunk.batch["values"][0].item() == i * 2
        assert chunk.batch["values"][1].item() == i * 2 + 1
    print("✓ Test Case 11 passed")


def test_chunk_large_chunk_count_not_divisible():
    """Test Case 12: Non-divisible large chunk count - 23/10."""
    print("=== Test Case 12: 23/10 (non-divisible large chunk count) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(23)}, meta_info={"test": "case12"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(10)
    assert len(chunks) == 10
    # First 3 chunks have 3 elements, last 7 chunks have 2 elements (with padding)
    assert len(chunks[0].batch["values"]) == 3
    assert len(chunks[1].batch["values"]) == 3
    assert len(chunks[2].batch["values"]) == 3
    assert len(chunks[3].batch["values"]) == 2
    assert len(chunks[4].batch["values"]) == 2
    assert len(chunks[5].batch["values"]) == 2
    assert len(chunks[6].batch["values"]) == 2
    assert len(chunks[7].batch["values"]) == 2
    assert len(chunks[8].batch["values"]) == 2
    assert len(chunks[9].batch["values"]) == 2
    print("✓ Test Case 12 passed")


def test_chunk_float_tensor():
    """Test Case 13: Float tensor - 15/3."""
    print("=== Test Case 13: 15/3 (float tensor) ===")
    data = DataProto.from_dict(tensors={"values": torch.randn(15)}, meta_info={"test": "case13"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(3)
    assert len(chunks) == 3
    assert len(chunks[0].batch["values"]) == 5
    assert len(chunks[1].batch["values"]) == 5
    assert len(chunks[2].batch["values"]) == 5
    print("✓ Test Case 13 passed")


def test_chunk_bool_tensor():
    """Test Case 14: Boolean tensor - 7/2."""
    print("=== Test Case 14: 7/2 (boolean tensor) ===")
    data = DataProto.from_dict(
        tensors={"values": torch.tensor([True, False, True, False, True, False, True])}, meta_info={"test": "case14"}
    )
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(2)
    assert len(chunks) == 2
    assert len(chunks[0].batch["values"]) == 4
    assert len(chunks[1].batch["values"]) == 3
    print("✓ Test Case 14 passed")


def test_chunk_long_tensor():
    """Test Case 15: Long integer tensor - 16/4."""
    print("=== Test Case 15: 16/4 (long integer tensor) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(16, dtype=torch.long)}, meta_info={"test": "case15"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk.batch["values"]) == 4
        assert chunk.batch["values"].dtype == torch.long
    print("✓ Test Case 15 passed")


def test_chunk_complex_meta_info():
    """Test Case 16: Complex meta_info - 12/3."""
    print("=== Test Case 16: 12/3 (complex meta_info) ===")
    complex_meta = {"test": "case16", "nested": {"key": "value", "list": [1, 2, 3]}, "array": np.array([1, 2, 3])}
    data = DataProto.from_dict(tensors={"values": torch.arange(12)}, meta_info=complex_meta)
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(3)
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.meta_info == complex_meta
    print("✓ Test Case 16 passed")


def test_chunk_multiple_non_tensor_fields():
    """Test Case 17: Multiple non_tensor fields - 6/2."""
    print("=== Test Case 17: 6/2 (multiple non_tensor fields) ===")
    data = DataProto.from_dict(
        tensors={"values": torch.arange(6)},
        non_tensors={"strings": ["a", "b", "c", "d", "e", "f"], "numbers": [1, 2, 3, 4, 5, 6]},
        meta_info={"test": "case17"},
    )
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(2)
    assert len(chunks) == 2
    assert chunks[0].non_tensor_batch["strings"].tolist() == ["a", "b", "c"]
    assert chunks[0].non_tensor_batch["numbers"].tolist() == [1, 2, 3]
    assert chunks[1].non_tensor_batch["strings"].tolist() == ["d", "e", "f"]
    assert chunks[1].non_tensor_batch["numbers"].tolist() == [4, 5, 6]
    print("✓ Test Case 17 passed")


def test_chunk_extreme_large_data():
    """Test Case 18: Extreme large data size - 1000/8."""
    print("=== Test Case 18: 1000/8 (extreme large data size) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(1000)}, meta_info={"test": "case18"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(8)
    assert len(chunks) == 8
    expected_size = 125
    for chunk in chunks:
        assert len(chunk.batch["values"]) == expected_size
    assert chunks[0].batch["values"][0].item() == 0
    assert chunks[7].batch["values"][-1].item() == 999
    print("✓ Test Case 18 passed")


def test_chunk_boundary_chunk_count():
    """Test Case 19: Boundary chunk count - 100/100."""
    print("=== Test Case 19: 100/100 (boundary chunk count) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(100)}, meta_info={"test": "case19"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(100)
    assert len(chunks) == 100
    for i, chunk in enumerate(chunks):
        assert len(chunk.batch["values"]) == 1
        assert chunk.batch["values"][0].item() == i
    print("✓ Test Case 19 passed")


if __name__ == "__main__":
    # Run all tests
    print("Starting comprehensive tests for DataProto.chunk method...")

    test_chunk_basic_divisible()
    test_chunk_not_divisible()
    test_chunk_data_less_than_chunks()
    test_chunk_single_data()
    test_chunk_large_data()
    test_chunk_large_data_not_divisible()
    test_chunk_multi_dimensional()
    test_chunk_multiple_tensors()
    test_chunk_with_non_tensor_batch()
    test_chunk_empty_data()
    test_chunk_large_chunk_count()
    test_chunk_large_chunk_count_not_divisible()
    test_chunk_float_tensor()
    test_chunk_bool_tensor()
    test_chunk_long_tensor()
    test_chunk_complex_meta_info()
    test_chunk_multiple_non_tensor_fields()
    test_chunk_extreme_large_data()
    test_chunk_boundary_chunk_count()

    print("\nAll 19 test cases passed!")
