import numpy as np
def quantize_simple(embeds1, embeds2, bits=8):
    """Simple quantization to fixed bit precision"""
    assert np.all(embeds1 >= 0)
    assert np.all(embeds2 >= 0)
    min_val = min(np.min(embeds1), np.min(embeds2))
    max_val = max(np.max(embeds1), np.max(embeds2))
    scale = (2**bits - 1) / (max_val - min_val)

    dtype = np.uint8 if bits <= 8 else np.uint16
    # dtype = np.int8 if bits <= 8 else np.int16
    max_qval = 2**bits - 1

    quantized1 = np.clip(np.round(embeds1  * scale), 0, max_qval).astype(dtype)
    quantized2 = np.clip(np.round(embeds2  * scale), 0, max_qval).astype(dtype)
    # quantized1 = np.round(embeds1  * scale).astype(dtype)
    # quantized2 = np.round(embeds2  * scale).astype(dtype)

    return quantized1, quantized2, scale

def quantize_per_feature(embeds1, embeds2, bits=8):
    assert np.all(embeds1 >= 0)
    assert np.all(embeds2 >= 0)
    # Stack both arrays vertically: shape (N1 + N2, D)
    embeddings = np.vstack((embeds1, embeds2))

    # Min/max per feature (column)
    min_vals = np.min(embeddings, axis=0)
    max_vals = np.max(embeddings, axis=0)

    # Compute scale per feature
    scale = (2**bits - 1) / (max_vals - min_vals)
    
    # Avoid division by zero
    constant_features = (max_vals == min_vals)
    scale[constant_features] = 1.0

    # Quantize
    quantized1 = np.round(embeds1 * scale)
    quantized2 = np.round(embeds2 * scale)

    # Clip and cast to appropriate type
    max_val = 2**bits - 1
    dtype = np.uint8 if bits <= 8 else np.uint16
    quantized1 = np.clip(quantized1, 0, max_val).astype(dtype)
    quantized2 = np.clip(quantized2, 0, max_val).astype(dtype)

    return quantized1, quantized2, scale
