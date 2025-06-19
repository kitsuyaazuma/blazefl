from typing import Any

import torch


def move_tensor_to_shared_memory(obj: Any, max_depth: int = 1) -> None:
    visited = set()

    def _recursive_helper(current_obj: Any, depth: int):
        if depth >= max_depth:
            return

        if isinstance(current_obj, dict | list | tuple) or hasattr(
            current_obj, "__dict__"
        ):
            obj_id = id(current_obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

        if isinstance(current_obj, torch.Tensor):
            current_obj.share_memory_()
        elif isinstance(current_obj, dict):
            for v in current_obj.values():
                _recursive_helper(v, depth + 1)
        elif isinstance(current_obj, list | tuple):
            for item in current_obj:
                _recursive_helper(item, depth + 1)
        elif hasattr(current_obj, "__dict__"):
            for v in current_obj.__dict__.values():
                _recursive_helper(v, depth + 1)

    _recursive_helper(obj, 0)
