from concrete import fhe
import numpy as np
from src.dataloader import DataLoader
from src.helpers import generate_inputset, quantize_simple, get_roc_metrics, distance

@fhe.compiler({"input1": "encrypted", "input2": "encrypted"})
def euclidian_distance(input1, input2):
    return np.sum((input1 - input2) ** 2)


compression = True
configuration = fhe.Configuration(
    compress_evaluation_keys=compression,
    compress_input_ciphertexts=compression,
    show_graph=True,
    # p_error = 1 / 10_000,
)

dimensions = 44
bw = 4
loader = DataLoader()
embeds1, embeds2, issame_list = loader.load_pairs(
    "./data/pair_embeddings_ceci.npz", dimensions=dimensions)

embed_shape = loader.get_embed_shape()
quantized1, quantized2, scale = quantize_simple(embeds1, embeds2, bw)

threshold = get_roc_metrics(distance(quantized1, quantized2), issame_list)
print(threshold)

inputset = generate_inputset(20,2**bw -1,embed_shape)

circuit = euclidian_distance.compile(inputset, configuration=configuration) #type: ignore
val = circuit.graph.maximum_integer_bit_width()
print("\t ==> Maximum integer bit width: ", val)
circuit.server.save("./server.zip")
print("Server file saved")
