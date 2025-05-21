import numpy as np
import pandas as pd
from concrete import fhe
from rich.console import Console
from rich.progress import Progress
from dataloader import DataLoader
import time

def define_euclidean_circuit():
    @fhe.compiler({"input1": "encrypted", "input2": "encrypted"})
    def euclidean_distance(input1, input2):
        return np.sum((input1 - input2) ** 2)
    return euclidean_distance

def generate_inputset(size,max_val_int, vector_dim, min_val = 0):
    return [ ([np.random.randint(min_val, max_val_int+1) for _ in range(vector_dim)],
            [np.random.randint(min_val,max_val_int+1) for _ in range(vector_dim)])
            for _ in range(size)
            ]

def sample_balanced_indices(labels, n_samples_per_class=50):
    # Convert to numpy array if it's not already
    labels = np.array(labels)
    
    # Find indices of True and False values
    true_indices = np.where(labels == True)[0]
    false_indices = np.where(labels == False)[0]
    
    # Check if we have enough samples of each class
    if len(true_indices) < n_samples_per_class:
        raise ValueError(f"Not enough True samples. Requested {n_samples_per_class}, but only {len(true_indices)} available")
    if len(false_indices) < n_samples_per_class:
        raise ValueError(f"Not enough False samples. Requested {n_samples_per_class}, but only {len(false_indices)} available")
    
    # Randomly sample from each class
    sampled_true_indices = np.random.choice(true_indices, size=n_samples_per_class, replace=False)
    sampled_false_indices = np.random.choice(false_indices, size=n_samples_per_class, replace=False)
    
    # Combine and return
    combined_indices = np.concatenate([sampled_true_indices, sampled_false_indices])
    
    # Optional: shuffle the combined indices
    np.random.shuffle(combined_indices)
    
    return combined_indices.tolist()


def quantize_simple(embeds1, embeds2, bits=8, signed=False):
    min_val = min(np.min(embeds1), np.min(embeds2))
    max_val = max(np.max(embeds1), np.max(embeds2))
    scale = (2**bits - 1) / (max_val - min_val)
    max_qval = 2**bits - 1
    if signed:
        print("signed")
        dtype = np.int8 if bits <= 8 else np.int16
        quantized1 = np.round(embeds1  * scale).astype(dtype)
        quantized2 = np.round(embeds2  * scale).astype(dtype)
    else:
        print("unsigned")
        assert np.all(embeds1 >= 0)
        assert np.all(embeds2 >= 0)
        dtype = np.uint8 if bits <= 8 else np.uint16
        quantized1 = np.clip(np.round(embeds1  * scale), 0, max_qval).astype(dtype)
        quantized2 = np.clip(np.round(embeds2  * scale), 0, max_qval).astype(dtype)
    return quantized1, quantized2, scale

console = Console()


def euclidian_clear(x, y):
    return np.sum((np.array(x) - np.array(y)) ** 2)

dimensions = [i for i in range(40, 45)]
dimensions = [44]
bits = [2, 3, 4, 5]
bits = [4]
num_pairs = 10
results = []
minmax = True 

for dim in dimensions:
    loader = DataLoader()
    embeds1, embeds2, issame_list = loader.load_pairs(
        "./data/pair_embeddings_ceci.npz", dimensions=dim, minmax=minmax)
    embed_shape = loader.get_embed_shape()

    indices = sample_balanced_indices(issame_list, num_pairs)

    with Progress() as progress:
        task = progress.add_task(
            f"[bold yellow]Running circuit at dim={dim}...[/bold yellow]",
            total=len(bits) * num_pairs*2)

        for bit in bits:

            if minmax: 
                assert np.all(embeds1 >= 0)
                assert np.all(embeds2 >= 0)
                print("minmax")
                quantized1, quantized2, scale = quantize_simple(embeds1, embeds2, bit, False)
            else:
                print("no minmax")
                quantized1, quantized2, scale = quantize_simple(embeds1, embeds2, bit, True)

            max_val = max(np.max(quantized1), np.max(quantized2))
            min_val = min(np.min(quantized1), np.min(quantized2))

            inputset = generate_inputset(5,max_val, embed_shape, min_val)

            compression = True
            configuration = fhe.Configuration(
                compress_evaluation_keys=compression,
                compress_input_ciphertexts=compression,
            )

            euclidean_distance = define_euclidean_circuit()
            circuit = euclidean_distance.compile(inputset, # type: ignore
                                                 configuration=configuration) 
            print("PBS COUNT: ", circuit.programmable_bootstrap_count)
            print("Complexity: ", circuit.complexity)
            mbw = circuit.graph.maximum_integer_bit_width()
            print("Dimensions: ", dim, "per feature BW: ", bit, "MBW: ", mbw)
            circuit.keygen()

            times = []
            for i in indices:
                e1, e2 = quantized1[i], quantized2[i]

                sample = (np.array(e1), np.array(e2))
                
                encrypted_e1, encrypted_e2 = circuit.encrypt(*sample) #type: ignore

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
                "mean": times.mean(),
                "std": times.std()
            }
            results.append(result_row)

            print("-----------------------")
            # print(f"For {p_error}")
            print(f"Mean time to calculate 1:1 match: {times.mean()} std: {times.std()}")

# Save to CSV
# df = pd.DataFrame(results)
# df.to_csv("timings.csv", index=False)
# console.print("[green bold]Results saved[/green bold]")
