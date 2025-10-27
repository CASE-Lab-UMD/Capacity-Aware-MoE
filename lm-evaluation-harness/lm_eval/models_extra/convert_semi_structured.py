import torch
from torch import nn
from .semi_structured import to_sparse_semi_structured, SparseSemiStructuredTensor

# SparseSemiStructuredTensor._FORCE_CUTLASS = False
SparseSemiStructuredTensor._FORCE_CUTLASS = True

# @torch.compile
def to_sparse_semi_structured_compiled(x):
    return to_sparse_semi_structured(x)

def is_semi_structured_weight(weight: torch.Tensor) -> bool:
    """🔍 Check if the weight is semi-structured (1:2 or 2:4)."""
    # Ensure the tensor is 2D.
    return True

    if weight.dim() != 2:
        return False

    # Check sparsity
    mask = (weight == 0)
    if mask.sum().item() != weight.numel() // 2:
        return False

    # Check 1:2
    is_1_2 = True
    sparse_param_count_1_2 = torch.full((weight.shape[1],), 1, dtype=torch.int64, device=weight.device)
    for row_group in mask.split(2, dim=0):
        # mask: shape(output_dim, input_dim)
        # row_group: shape(2, input_dim)
        if not torch.equal(row_group.sum(dim=0), sparse_param_count_1_2):
            is_1_2 = False
            break
    if is_1_2:
        return True

    # Check 2:4
    is_2_4 = True
    sparse_param_count_2_4 = torch.full((weight.shape[1],), 2, dtype=torch.int64, device=weight.device)
    for row_group in mask.split(4, dim=0):
        # mask: shape(output_dim, input_dim)
        # row_group: shape(4, input_dim)
        if not torch.equal(row_group.sum(dim=0), sparse_param_count_2_4):
            is_2_4 = False
            break
    if is_2_4:
        return True

    return False


def convert_semi_structured_weights(model):
    # 🔍 Automatically check & convert semi-structured sparse weights in the model
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear)):
            if "experts" in name: 
                # print(f"name: {name}, device: {devoce}")
                if is_semi_structured_weight(module.weight.data):
                    device = module.weight.device
                    # module.weight.to("cpu")
                    # weight = module.weight
                    # del weight
                    # torch.cuda.empty_cache()

                    # module.weight = nn.Parameter(to_sparse_semi_structured(module.weight))
                    # module.weight = nn.Parameter(to_sparse_semi_structured_compiled(module.weight))
                    # mask = (module.weight.data != 0).bool()
                    # module.weight = nn.Parameter(mask * module.weight)
                    # weight = to_sparse_semi_structured_compiled(module.weight)
                    # module.weight.to("cpu")
                    # del module.weight
                    with torch.inference_mode():
                        module.weight = nn.Parameter(to_sparse_semi_structured_compiled(module.weight))
                    # # .to(device) # only support 2:4 now. 
                    print(f"module.weight.requires_grad: {module.weight.requires_grad}")
                    print(f"Converted {name} weights to semi-structured.")                
                    torch.cuda.empty_cache()
                    mem = torch.cuda.memory_allocated() / (1024 ** 2)
                    print(f"Mem: {mem:.3f}MB")
                    # break
        
