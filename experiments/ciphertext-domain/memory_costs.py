import numpy as np
import pandas as pd
from concrete import fhe
from rich.console import Console
from rich.progress import Progress
from dataloader import DataLoader
from helpers import generate_inputset, sample_balanced_indices

from quantizers import quantize_simple 

import time

console = Console()

def define_euclidean_circuit():
    @fhe.compiler({"input1": "encrypted", "input2": "encrypted"})
    def euclidean_distance(input1, input2):
        return np.sum((input1 - input2) ** 2)
    return euclidean_distance

def euclidian_clear(x, y):
    return np.sum((np.array(x) - np.array(y)) ** 2)

bit = 5
dim = 44
num_pairs = 2
results = []
compression = True 
loader = DataLoader()
embeds1, embeds2, issame_list = loader.load_pairs(
    "./data/pair_embeddings_ceci.npz", dimensions=dim)
embed_shape = loader.get_embed_shape()

indices = sample_balanced_indices(issame_list, num_pairs)

with Progress() as progress:
    task = progress.add_task(
        f"[bold yellow]Running circuit at dim={dim}...[/bold yellow]",
        total=num_pairs*2)

    quantized1, quantized2, _ = quantize_simple(embeds1, embeds2, bit)

    max_val = max(np.max(quantized1), np.max(quantized2))
    min_val = min(np.min(quantized1), np.min(quantized2))
    inputset = generate_inputset(50, max_val, embed_shape, min_val)

    configuration = fhe.Configuration(
        compress_evaluation_keys=compression,
        compress_input_ciphertexts=compression,
        show_graph=True,
        )
    

    euclidean_distance = define_euclidean_circuit()
    circuit = euclidean_distance.compile(inputset, configuration=configuration) # type: ignore

    print("PBS COUNT: ", circuit.programmable_bootstrap_count)
    print("Complexity: ", circuit.complexity)
    circuit.keygen()
    print(f"Size of the evaluation keys {len(circuit.keys.serialize())/(1024**2)}")
    mbw = circuit.graph.maximum_integer_bit_width()

    times = []
    ct_sizes = []
    for i in indices:
        e1, e2 = quantized1[i], quantized2[i]

        sample = (np.array(e1), np.array(e2))
        
        encrypted_e1, encrypted_e2 = circuit.encrypt(*sample) #type: ignore
        print(
            f"Size of ciphertexts {len(encrypted_e1.serialize())/1024} , \
            {len(encrypted_e2.serialize())}")
        ct_sizes.append(len(encrypted_e1.serialize())/1024)

        start_time = time.time()
        encrypted_result = circuit.run(encrypted_e1, encrypted_e2)

        end_time = time.time()
        times.append(end_time-start_time)
        result_int = circuit.decrypt(encrypted_result)
        # print(result_int, euclidian_clear(e1,e2))
        progress.update(task, advance=1)

    times = np.array(times)
    result_row = {
        "dimension": dim,
        "bit_width": bit,
        "MBW": mbw,
        "compression": "True" if compression else "False",
        "mean": times.mean(),
        "std": times.std(),
        "eval_key_sizes_mb":len(circuit.keys.serialize())/(1024**2),
        "ct_sizes_kb":np.mean(ct_sizes)
    }
    results.append(result_row)

    print("-----------------------")
    print("Dimensions: ", dim, "per feature BW: ", bit, "MBW: ", mbw)
    print(f"Mean time to calculate 1:1 match: {times.mean()} std: {times.std()}")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("./ciphertext-domain/results/enc_memory_costs.csv", index=False)
console.print("[green bold]Results saved[/green bold]")
