"""Keep the attention integration outside Dynamo graphs."""


def flash_attn_dynamo_disable_wrapper(original, options):
    import torch
    return torch._dynamo.disable(original)
