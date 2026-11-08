## 2024-05-24 - Device type checking in PyTorch
**Learning:** Checking `getattr(device, 'type', '')` is dangerous because PyTorch device arguments can frequently be passed as integers (e.g. `0`) or strings (e.g. `'cuda:0'`), neither of which have a `type` attribute, leading to silent failures when identifying CUDA devices.
**Action:** When validating PyTorch device identifiers, always check `isinstance(device, int) or str(device).startswith('cuda')` to safely handle integers, strings, and torch.device objects.
## 2024-06-03 - PyTorch Tensor Reallocation in Hot Loops
**Learning:** Hardcoded arrays converted to `torch.tensor` inside frequently called functions (like `vae_decode_fake` used for video frame previews) cause significant performance overhead due to constant memory reallocation and CPU-to-GPU transfers.
**Action:** Extract static data to module level and implement a dictionary cache keyed by `(device, dtype)` to reuse tensors. This avoids repeated overhead while supporting dynamic device placement.

## 2024-06-03 - VRAM Polling Overhead in Model Iteration
**Learning:** Calling `torch.cuda.memory_stats()` or `torch.cuda.mem_get_info()` continuously during module iteration (like moving a large model with `model.modules()`) introduces a massive bottleneck. The CUDA runtime syncs the CPU and GPU on each check, causing a massive stall.
**Action:** When tracking VRAM floor/ceiling limits dynamically, always batch the polling checks using a modulus (e.g. `if i % 25 == 0`) rather than checking on every single iteration step.

## 2024-07-07 - Numpy Poly1d Evaluation Overhead in Hot Loops
**Learning:** Evaluating `np.poly1d` on single scalar values inside PyTorch hot loops introduces significant Numpy overhead (~10-15 microseconds per call) due to type checking and wrapping.
**Action:** When a polynomial evaluation is needed on scalar values in a hot loop, extract the scalar using `.cpu().item()` and use an unrolled native Python lambda (via Horner's method) to reduce evaluation time to ~100 nanoseconds.
## 2024-07-19 - Vectorizing cu_seqlens loops
**Learning:** Python loops over batch sizes to calculate cumulative sequence lengths (cu_seqlens) introduce unnecessary CPU overhead, and hardcoding device="cuda" breaks execution on other devices like DML or CPU.
**Action:** When building cumulative lengths or offsets based on batch size, use vectorized PyTorch tensor slice assignments (`1::2`, `2::2`) and `torch.arange` on the input tensor's device to calculate them in one pass.
## 2024-07-22 - Implicit CPU-GPU Sync from Python Lists
**Learning:** Creating a PyTorch tensor directly from a Python list containing elements (or a scalar) and immediately moving it to a device (e.g., `torch.tensor([val] * bs).to(device)`) forces an implicit, blocking CPU-GPU synchronization overhead.
**Action:** When initializing tensors with constant values, use `torch.full()` or `torch.zeros()`/`torch.ones()` directly on the target device to avoid CPU-GPU synchronization and reduce overhead.

## 2024-07-23 - CPU-GPU Sync on Device Tensors
**Learning:** Calling `.sum()` on a GPU tensor and immediately converting it to a Python integer (e.g., `int(tensor.sum())` or `tensor.sum().item()`) forces a blocking CPU-GPU synchronization, functioning identically to `.item()`.
**Action:** Compute sums on the CPU before moving tensors to the device to avoid this overhead in hot loops or setup logic.

## 2024-07-25 - torch.stack vs torch.tensor Type Sensitivity
**Learning:** While `torch.stack(list)` avoids the severe CPU-GPU sync bottleneck introduced by `torch.tensor(list, device=device)`, it is highly type-sensitive and will instantly crash with a `TypeError` if the list contains any standard Python primitives (like floats or ints).
**Action:** When refactoring list-to-tensor operations to use `torch.stack()` for performance, ensure every element in the list is explicitly cast to a tensor (e.g., `torch.tensor(1., device=device)`) before calling `.stack()`.

## 2024-07-26 - Dynamic Tuple Indexing Overhead in PyTorch Hot Loops
**Learning:** Dynamically creating shape tuples for broadcasting (e.g., `(...,) + (None,) * (dims - 1)`) inside hot loops introduces PyTorch CPU overhead. In PyTorch, indexing with `...` and `None` requires the library to parse the complex tuple and route it through the advanced indexing dispatcher.
**Action:** Pre-calculate the shape and use `.reshape(shape)` instead to improve execution speed safely. Avoid using `.view()` as it will throw a RuntimeError on non-contiguous tensors. `.reshape()` safely handles non-contiguous tensors and avoids unsafe flattening while maintaining speed improvements over dynamic tuple indexing.

## 2024-10-31 - Redundant CPU Transfer
**Learning:** Calling `.cpu()` before `.item()` on a GPU tensor introduces unnecessary memory allocation overhead because `.item()` already handles the host transfer.
**Action:** Call `.item()` directly on the tensor.

## 2024-05-18 - Avoiding implicit CPU-GPU synchronization in UniPC sampler
**Learning:** Creating a PyTorch tensor from a Python list of GPU tensors (or mixed with floats) like `torch.tensor(rks)` inside the sampler loop forces an implicit, blocking CPU-GPU synchronization because PyTorch moves the data to the CPU to assemble it.
**Action:** Always use `torch.stack()` (and ensure all elements are tensors, not floats) instead of `torch.tensor()` for lists of tensors in hot loops to avoid device synchronization overhead.
