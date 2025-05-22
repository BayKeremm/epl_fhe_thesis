## Communication Scenario 
In this directory we  describe a practical communication scenario that illustrates how FHE can be integrated alongside traditional cryptographic tools.

The client-server architecture is used.  Where the client would like  to offload the biometric data  comparison to the server without  revealing the data. This is where FHE  is used, it allows us to compute useful functions on encrypted data and send back results in encrypted domain  where only the client can decrypt. 

The following interaction steps are  implemented: 

- Client Introduce
    - The client and server perform an ECDH exchange on the `secp256k1` curve to derive a shared secret. This secret is then expanded using HKDF into a session key $K$, which is then bound to a client ID. The server also records $(ID, Q_c)$ for later use.
- Schnorr Identification
    - To prove possession of the private key corresponding to the stored public key $Q_c$ under client ID, the client engages in Schnorr's zero-knowledge identification protocol. Upon successful verification of the proof, the server generates a challenge message by concatenating a freshly generated nonce and timestamp. This challenge is stored and sent to the client, bound to the client's ID.

- Client Enroll
    - Using the session channel, the client transmits its encrypted face embeddings $\mathbf{\vec E}$ and the corresponding TFHE evaluation keys $\mathbf{\vec K_e}$, and a confirmation in the form of an HMAC-SHA256 over the challenge message. The server verifies the HMAC confirmation, then stores $\{\mathbf{\vec E},\mathbf{\vec K_e}\}$ under the client’s $ID$.

- Client Authenticate
    - For each authentication attempt, the client repeats the Schnorr identification to obtain a fresh challenge message. It then sends new encrypted face embeddings $\mathbf{\vec E}^\*$ and the HMAC confirmation. The server verifies the confirmation, homomorphically compares $\mathbf{\vec E}$ and $\mathbf{\vec E}^\*$ using its stored TFHE evaluation keys under client's ID, and returns the result of the comparison.

## How to run the code?
Setup the environment, `concrete` works only with python 3.9 to 3.12 inclusive. 
```
python -m venv comm-venv
source comm-venv/bin/activate
pip install -r requirements.txt
```

First the trusted party runs the following to generate server files:

```
PYTHONPATH=. python src/trusted.py
```

Which generates `server.zip`. 

Then we can run the client and the server in different terminals:
```
PYTHONPATH=. python src/server.py 
PYTHONPATH=. python src/client.py 
```
Then follow the menu on the client to interact with the server.
