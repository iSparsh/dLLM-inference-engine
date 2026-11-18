"""Constants shared by both reference models."""

MODEL_IDS = {
    "dllm": "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    "ar": "Qwen/Qwen2.5-1.5B-Instruct",
}
MASK, EOS, BOS = 151665, 151645, 151643
BLOCK, SUB_BLOCK = 32, 8
LAYERS, HIDDEN, INTERMEDIATE, VOCAB = 28, 1536, 8960, 151936
Q_HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128
THETA, EPS = 1_000_000.0, 1e-6
