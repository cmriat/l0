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

"""测试DataProto.chunk方法的各种场景和边界情况."""

from l0.verl_adapter import monkey_patch_tensordict_chunk

monkey_patch_tensordict_chunk()
import numpy as np
import torch
from verl.protocol import DataProto


def test_chunk_basic_divisible():
    """测试用例1: 基本整除情况 - 8/4."""
    print("=== 测试用例1: 8/4 (整除) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(8)}, meta_info={"test": "case1"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert chunks[0].batch["values"].tolist() == [0, 1]
    assert chunks[1].batch["values"].tolist() == [2, 3]
    assert chunks[2].batch["values"].tolist() == [4, 5]
    assert chunks[3].batch["values"].tolist() == [6, 7]
    print("✓ 测试用例1通过")


def test_chunk_not_divisible():
    """测试用例2: 不能整除情况 - 9/4."""
    print("=== 测试用例2: 9/4 (不能整除) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(9)}, meta_info={"test": "case2"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert chunks[0].batch["values"].tolist() == [0, 1, 2]
    assert chunks[1].batch["values"].tolist() == [3, 4]
    assert chunks[2].batch["values"].tolist() == [5, 6]
    assert chunks[3].batch["values"].tolist() == [7, 8]
    print("✓ 测试用例2通过")


def test_chunk_data_less_than_chunks():
    """测试用例3: 数据量小于chunk数 - 3/4."""
    print("=== 测试用例3: 3/4 (数据量小于chunk数) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(3)}, meta_info={"test": "case3"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert chunks[0].batch["values"].tolist() == [0]
    assert chunks[1].batch["values"].tolist() == [1]
    assert chunks[2].batch["values"].tolist() == [2]
    assert chunks[3].batch["values"].tolist() == []  # 自动填充
    print("✓ 测试用例3通过")


def test_chunk_single_data():
    """测试用例4: 单个数据 - 1/4."""
    print("=== 测试用例4: 1/4 (单个数据) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(1)}, meta_info={"test": "case4"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert chunks[0].batch["values"].tolist() == [0]
    assert chunks[1].batch["values"].tolist() == []  # 自动填充
    assert chunks[2].batch["values"].tolist() == []  # 自动填充
    assert chunks[3].batch["values"].tolist() == []  # 自动填充
    print("✓ 测试用例4通过")


def test_chunk_large_data():
    """测试用例5: 大数据量 - 100/4."""
    print("=== 测试用例5: 100/4 (大数据量) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(100)}, meta_info={"test": "case5"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    assert len(chunks[0].batch["values"]) == 25
    assert chunks[0].batch["values"][0].item() == 0
    assert chunks[3].batch["values"][-1].item() == 99
    print("✓ 测试用例5通过")


def test_chunk_large_data_not_divisible():
    """测试用例6: 大数据量不能整除 - 101/4."""
    print("=== 测试用例6: 101/4 (大数据量不能整除) ===")
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
    print("✓ 测试用例6通过")


def test_chunk_multi_dimensional():
    """测试用例7: 多维度tensor - 12/3."""
    print("=== 测试用例7: 12/3 (多维度tensor) ===")
    data = DataProto.from_dict(tensors={"values": torch.randn(12, 5, 3)}, meta_info={"test": "case7"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(3)
    assert len(chunks) == 3
    assert chunks[0].batch["values"].shape == (4, 5, 3)
    assert chunks[1].batch["values"].shape == (4, 5, 3)
    assert chunks[2].batch["values"].shape == (4, 5, 3)
    print("✓ 测试用例7通过")


def test_chunk_multiple_tensors():
    """测试用例8: 多个tensor字段 - 10/2."""
    print("=== 测试用例8: 10/2 (多个tensor字段) ===")
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
    print("✓ 测试用例8通过")


def test_chunk_with_non_tensor_batch():
    """测试用例9: 包含non_tensor_batch - 8/4."""
    print("=== 测试用例9: 8/4 (包含non_tensor_batch) ===")
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
    print("✓ 测试用例9通过")


def test_chunk_empty_data():
    """测试用例10: 边界情况 - 0/4."""
    print("=== 测试用例10: 0/4 (空数据) ===")
    data = DataProto.from_dict(tensors={"values": torch.empty(0)}, meta_info={"test": "case10"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk.batch["values"]) == 0
    print("✓ 测试用例10通过")


def test_chunk_large_chunk_count():
    """测试用例11: 大chunk数 - 20/10."""
    print("=== 测试用例11: 20/10 (大chunk数) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(20)}, meta_info={"test": "case11"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(10)
    assert len(chunks) == 10
    for i, chunk in enumerate(chunks):
        assert len(chunk.batch["values"]) == 2
        assert chunk.batch["values"][0].item() == i * 2
        assert chunk.batch["values"][1].item() == i * 2 + 1
    print("✓ 测试用例11通过")


def test_chunk_large_chunk_count_not_divisible():
    """测试用例12: 不能整除的大chunk数 - 23/10."""
    print("=== 测试用例12: 23/10 (不能整除的大chunk数) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(23)}, meta_info={"test": "case12"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(10)
    assert len(chunks) == 10
    # 前3个chunk有3个元素，后7个chunk有2个元素（加上填充）
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
    print("✓ 测试用例12通过")


def test_chunk_float_tensor():
    """测试用例13: 浮点数tensor - 15/3."""
    print("=== 测试用例13: 15/3 (浮点数tensor) ===")
    data = DataProto.from_dict(tensors={"values": torch.randn(15)}, meta_info={"test": "case13"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(3)
    assert len(chunks) == 3
    assert len(chunks[0].batch["values"]) == 5
    assert len(chunks[1].batch["values"]) == 5
    assert len(chunks[2].batch["values"]) == 5
    print("✓ 测试用例13通过")


def test_chunk_bool_tensor():
    """测试用例14: 布尔tensor - 7/2."""
    print("=== 测试用例14: 7/2 (布尔tensor) ===")
    data = DataProto.from_dict(
        tensors={"values": torch.tensor([True, False, True, False, True, False, True])}, meta_info={"test": "case14"}
    )
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(2)
    assert len(chunks) == 2
    assert len(chunks[0].batch["values"]) == 4
    assert len(chunks[1].batch["values"]) == 3
    print("✓ 测试用例14通过")


def test_chunk_long_tensor():
    """测试用例15: 长整型tensor - 16/4."""
    print("=== 测试用例15: 16/4 (长整型tensor) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(16, dtype=torch.long)}, meta_info={"test": "case15"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(4)
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk.batch["values"]) == 4
        assert chunk.batch["values"].dtype == torch.long
    print("✓ 测试用例15通过")


def test_chunk_complex_meta_info():
    """测试用例16: 复杂meta_info - 12/3."""
    print("=== 测试用例16: 12/3 (复杂meta_info) ===")
    complex_meta = {"test": "case16", "nested": {"key": "value", "list": [1, 2, 3]}, "array": np.array([1, 2, 3])}
    data = DataProto.from_dict(tensors={"values": torch.arange(12)}, meta_info=complex_meta)
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(3)
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.meta_info == complex_meta
    print("✓ 测试用例16通过")


def test_chunk_multiple_non_tensor_fields():
    """测试用例17: 多个non_tensor字段 - 6/2."""
    print("=== 测试用例17: 6/2 (多个non_tensor字段) ===")
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
    print("✓ 测试用例17通过")


def test_chunk_extreme_large_data():
    """测试用例18: 极大数据量 - 1000/8."""
    print("=== 测试用例18: 1000/8 (极大数据量) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(1000)}, meta_info={"test": "case18"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(8)
    assert len(chunks) == 8
    expected_size = 125
    for chunk in chunks:
        assert len(chunk.batch["values"]) == expected_size
    assert chunks[0].batch["values"][0].item() == 0
    assert chunks[7].batch["values"][-1].item() == 999
    print("✓ 测试用例18通过")


def test_chunk_boundary_chunk_count():
    """测试用例19: 边界chunk数 - 100/100."""
    print("=== 测试用例19: 100/100 (边界chunk数) ===")
    data = DataProto.from_dict(tensors={"values": torch.arange(100)}, meta_info={"test": "case19"})
    data.meta_info["_verl_auto_padding"] = True
    chunks = data.chunk(100)
    assert len(chunks) == 100
    for i, chunk in enumerate(chunks):
        assert len(chunk.batch["values"]) == 1
        assert chunk.batch["values"][0].item() == i
    print("✓ 测试用例19通过")


if __name__ == "__main__":
    # 运行所有测试
    print("开始全面测试DataProto.chunk方法...")

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

    print("\n所有19个测试用例全部通过！")
