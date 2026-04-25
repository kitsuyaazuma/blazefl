from copy import copy
from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class SHMHandle:
    """
    Sentinel placeholder for a tensor stored in shared memory.

    ``process_tensors_in_object(..., mode="replace")`` swaps ``torch.Tensor``
    instances for ``SHMHandle()`` so a lightweight, serializable object graph can
    be sent between processes without copying tensor payloads. During
    reconstruction, ``reconstruct_from_shared_memory`` treats this empty marker as
    "take the corresponding tensor from the shared-memory buffer".
    """

    pass


def process_tensors_in_object[T](
    obj: T, mode: Literal["move", "replace"], max_depth: int = 10
) -> T:
    """
    Recursively traverses an object to process `torch.Tensor` instances.

    This function handles different data structures, including dictionaries,
    lists, tuples, and general objects with a `__dict__` attribute. It also
    manages circular references to prevent infinite recursion.

    Args:
        obj: The object to process.
        mode:
            - "move": Moves tensors to shared memory in-place by calling
                      `.share_memory_()`. This is suitable for sending data from a
                      parent to a worker process.
            - "replace": Creates a deep copy of the object, replacing all
                         tensors with a lightweight `SHMHandle()`. This is used
                         to create a serializable "receipt" object that can be
                         returned from a worker to the parent without copying
                         tensor data.
        max_depth: The maximum recursion depth. Defaults to 10.

    Returns:
        - In "move" mode, returns the original object, modified in-place.
        - In "replace" mode, returns a new object where tensors have been
          replaced with `SHMHandle` instances.
    """
    # memo is used in 'replace' mode to handle circular references, similar to
    # copy.deepcopy.
    memo: dict[int, Any] = {}
    # visited is used in 'move' mode to track objects already processed.
    visited = set()

    def _recursive_helper(current_obj: Any, depth: int) -> Any:
        obj_id = id(current_obj)
        # Handle circular references based on the current mode.
        if mode == "move":
            if obj_id in visited:
                return
            visited.add(obj_id)
        elif mode == "replace":
            if obj_id in memo:
                return memo[obj_id]

        # Base case: Process torch.Tensor objects.
        if isinstance(current_obj, torch.Tensor):
            if mode == "move":
                # Move the tensor to shared memory in-place.
                current_obj.share_memory_()
                return current_obj
            elif mode == "replace":
                # Replace the tensor with a handle.
                return SHMHandle()

        # Stop recursion if the maximum depth is reached.
        if depth >= max_depth:
            return current_obj

        # --- Recursive processing for container types ---

        # Handle lists and tuples.
        if isinstance(current_obj, list | tuple):
            if mode == "move":
                # Recursively process each item for in-place modification.
                for item in current_obj:
                    _recursive_helper(item, depth + 1)
                return
            elif mode == "replace":
                # Create a new list/tuple with processed items.
                new_list = [_recursive_helper(item, depth + 1) for item in current_obj]
                new_obj = type(current_obj)(new_list)
                memo[id(current_obj)] = new_obj
                return new_obj

        # Handle dictionaries.
        if isinstance(current_obj, dict):
            if mode == "move":
                for v in current_obj.values():
                    _recursive_helper(v, depth + 1)
                return
            elif mode == "replace":
                # Create a new dictionary with processed values.
                new_dict = {
                    k: _recursive_helper(v, depth + 1) for k, v in current_obj.items()
                }
                memo[id(current_obj)] = new_dict
                return new_dict

        # Handle general objects (including dataclasses).
        if hasattr(current_obj, "__dict__"):
            if mode == "move":
                # Recursively process the object's __dict__.
                _recursive_helper(current_obj.__dict__, depth + 1)
                return
            elif mode == "replace":
                # Create a shallow copy to avoid modifying the original object's
                # structure.
                new_obj = copy(current_obj)
                memo[id(current_obj)] = new_obj
                # Recursively process the __dict__ and update the new object.
                new_obj.__dict__.update(
                    _recursive_helper(current_obj.__dict__, depth + 1)
                )
                return new_obj

        # For non-container types in 'replace' mode, memoize and return them.
        if mode == "replace":
            memo[id(current_obj)] = current_obj
            return current_obj
        return

    # --- Initial call to the recursive helper ---
    if mode == "move":
        # Start the in-place process. The original object is modified.
        _recursive_helper(obj, 0)
        return obj
    elif mode == "replace":
        # Start the replacement process, which returns a new, modified object.
        return _recursive_helper(obj, 0)


def reconstruct_from_shared_memory[T](handle_obj: T, shm_obj: T) -> T:
    """
    Recursively reconstructs an object from a handle-based object and a
    shared memory buffer object.

    This function traverses both objects simultaneously. For each attribute,
    if the `handle_obj` contains an `SHMHandle`, it takes the corresponding tensor
    from the `shm_obj`. Otherwise, it takes the value from the `handle_obj`.
    This ensures that non-tensor data is also correctly restored.

    Args:
        handle_obj: The object containing SHMHandles as placeholders.
        shm_obj: The object containing the actual tensors in shared memory.

    Returns:
        A new, fully reconstructed object with tensors populated from shared memory.
    """
    memo: dict[int, Any] = {}

    def _recursive_reconstruct(h_obj: Any, s_obj: Any) -> Any:
        # Handle circular references using the memo dictionary.
        obj_id = id(h_obj)
        if obj_id in memo:
            return memo[obj_id]

        # If we find a handle, we take the corresponding value from the shm_obj.
        if isinstance(h_obj, SHMHandle):
            return s_obj

        # For containers, we reconstruct them recursively.
        if isinstance(h_obj, list | tuple):
            new_list = [
                _recursive_reconstruct(h_item, s_item)
                for h_item, s_item in zip(h_obj, s_obj, strict=True)
            ]
            new_container = type(h_obj)(new_list)
            memo[obj_id] = new_container
            return new_container

        if isinstance(h_obj, dict):
            new_dict = {
                key: _recursive_reconstruct(h_val, s_obj[key])
                for key, h_val in h_obj.items()
            }
            memo[obj_id] = new_dict
            return new_dict

        if hasattr(h_obj, "__dict__"):
            # Create a shallow copy to preserve the object's type and non-`__dict__`
            # attributes.
            new_obj = copy(h_obj)
            memo[obj_id] = new_obj
            # Recursively reconstruct the __dict__.
            new_obj.__dict__.update(
                _recursive_reconstruct(h_obj.__dict__, s_obj.__dict__)
            )
            return new_obj

        # For all other types (primitives, etc.), return the value from the handle
        # object.
        return h_obj

    return _recursive_reconstruct(handle_obj, shm_obj)


def serialize_model(model: torch.nn.Module, cpu: bool = True) -> torch.Tensor:
    """
    Serialize a PyTorch model's parameters into a flat tensor.

    Args:
        model (torch.nn.Module): The PyTorch model to serialize.
        cpu (bool): Whether to move the serialized parameters to the CPU.

    Returns:
        torch.Tensor: A flat tensor containing the serialized parameters.
    """
    parameters = [param.data.view(-1) for param in model.state_dict().values()]
    serialized_parameters = torch.cat(parameters)
    if cpu:
        serialized_parameters = serialized_parameters.cpu()

    return serialized_parameters


def deserialize_model(
    model: torch.nn.Module, serialized_parameters: torch.Tensor
) -> None:
    """
    Deserialize a flat tensor back into a PyTorch model's parameters.

    Args:
        model (torch.nn.Module): The PyTorch model to update.
        serialized_parameters (torch.Tensor): The tensor containing the parameters.

    Returns:
        None
    """
    current_index = 0
    for param in model.state_dict().values():
        numel = param.numel()
        size = param.size()
        param.copy_(
            serialized_parameters[current_index : current_index + numel].view(size)
        )
        current_index += numel
