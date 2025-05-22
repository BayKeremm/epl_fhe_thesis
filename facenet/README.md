# Producing FaceNet embeddings
In this directory we produce the 512-dimensional facenet embeddings. 
To do this, ceci clusters are used. First we create a python environment with the following:

```
python3.11 -m venv try-venv
source try-venv/bin/activate
pip install -r requirements.txt
```
Then we can use a GPU to run the code to produce the LFW embeddings:
```
srun --partition=gpu --gres=gpu:1 --mem-per-gpu=4096  python runner.py
```
Once it finishes, confirm the results and save pair embeddings with:
```
python eval.py
```
which should give the output:

```
Pairs embeddings shape: (12000, 512)
Labels shape: (6000,)
Mean accuracy 0.9934999999999998
Saved pair embeddings!
```
The pairs are saved to the file `pair_embeddings_ceci.npz` which contains `embeddings` and `issame_list` numpy arrays. `embeddings` array has shape $(12000,512)$ and `issame_list` has shape $(6000,)$. 
The `embeddings` array has in even indeces the first set of pairs and in odd indeces the second set of pairs. 

This pairs data are needed for the other parts of the code. 
