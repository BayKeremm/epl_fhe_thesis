import socket
import json
from concrete import fhe
import numpy as np
from crypto.EClib import Point
from src.helpers import preprocess, clear_console
from crypto.secp256k1 import secp256k1_curve as curve
from random import SystemRandom
from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA512
from Crypto.Hash import HMAC, SHA256
import base64
from collections import Counter
from rich.console import Console

class Client:
    def __init__(self, database_id, data_dir,
                 dims, num_imgs, bw=4,
                 server_host='localhost',
                 server_port=8888):

        self.server_host = server_host
        self.server_port = server_port
        self.socket = None
        self.console = Console()
        clear_console()

        self.console.print(f"[bold magenta]Reading data for id: {database_id} [/bold magenta]")
        assert num_imgs % 2 == 0
        data = np.load(data_dir)
        embeds = data["embeddings"]
        classes = data["classes"]
        processed, _ = preprocess(embeds_in=
                                    embeds,
                                    dim=dims, minmax=True)
        idxs = classes == database_id
        self.embeds = processed[idxs][:num_imgs]
        # print(self.embeds.shape)
        self.bw = bw

        self.console.print(f"[bold magenta]Creating sk,pk for secp256k1...[/bold magenta]")
        self.prg = SystemRandom()
        self.sk = self.prg.randrange(1,curve.generator.n-1)
        self.pk = self.sk * curve.generator.G

        self.server_pk = None
        self.shared_secret = None
        self.hasher = SHA512.new(data=b'first')
        self.costs = [0, 0, 0, 0]  # [introduce, schnorr, enroll, authenticate]

    def connect(self):
        self.socket = socket.socket(socket.AF_INET,
                                    socket.SOCK_STREAM)
        try:
            self.socket.connect((self.server_host,
                                 self.server_port))
            return True
        except socket.error as e:
            self.console.print(f"[bold red]Connection error: {e}[/bold red]")
            return False
    
    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        print(self.costs)
    
    def send_request(self, request, idx):
        if not self.socket:
            if not self.connect():
                return {"status": "error",
                        "message": "Not connected to server"}
        try:
            message = json.dumps(request) + "\n"  # Append newline as a delimiter
            self.socket.send(message.encode('utf-8'))  # type: ignore
            response_data = ""
            while True:
                chunk = self.socket.recv(4096).decode('utf-8')  # type: ignore
                if not chunk:
                    break
                response_data += chunk
                if "\n" in chunk:  # Stop reading once a complete message is received
                    break

            # print(type(response_data),len(response_data))
            self.costs[idx]+=len(message.encode('utf-8'))
            self.costs[idx]+=len(response_data.encode('utf-8'))

            return json.loads(response_data.strip())  # Ensure clean parsing
        except Exception as e:
            self.console.print(f"[bold red]Error handling client: {e}[/bold red]")
            self.close()
            return {"status": "error", "message": str(e)}

    def introduce(self):
        request = {
            "action": "introduce",
                "pk": [self.pk.x, self.pk.y]
        }
        cost = json.dumps(request)
        print(f"Introduce: Client sends: {len(cost.encode('utf-8'))}")
        response = self.send_request(request,0)
        cost = json.dumps(response)
        print(f"Introduce: Server sends: {len(cost.encode('utf-8'))}")
        # print(f"Response from server datatype {type(response)} with size {len(response)}")
        self.server_pk = Point(curve, response.get("pk")[0],response.get("pk")[1])# type: ignore
        received_specs = response["specs"]
        self.user_id = response.get("id")
        client_specs = fhe.ClientSpecs.deserialize(received_specs.encode())
        self.client = fhe.Client(client_specs)

        salt = response.get("salt")
        shared_secret = (self.sk * self.server_pk).x.to_bytes(32,"big") # type: ignore
        self.shared_secret =  HKDF(shared_secret, 32, salt.to_bytes(16,'big'), SHA512) # type: ignore
        self.console.print("Introduced")
        return response

    def perform_schnorr(self) -> bytes:
        x = self.prg.randint(1,curve.generator.n-1)
        I = x * curve.generator.G
        request = {
            "action": "client_hello",
            "I": [I.x,I.y],
            "claimed_id": self.user_id
        }

        cost = json.dumps(request)
        print(f"Schnorr: Client sends: {len(cost.encode('utf-8'))}")
        response = self.send_request(request,1)
        cost = json.dumps(response)
        print(f"Schnorr: Server sends: {len(cost.encode('utf-8'))}")
        # print(f"Challenge schnorr from server datatype {type(response)} with size {len(response)}")
        c = response.get("c")

        s = (x + c * self.sk) % curve.generator.n # type: ignore

        challenge_response = {
            "action": "client_response",
            "claimed_id":self.user_id,
            "s":s
        }
        cost = json.dumps(challenge_response)
        print(f"Schnorr: Client sends: {len(cost.encode('utf-8'))}")

        response = self.send_request(challenge_response,1)
        cost = json.dumps(response)
        print(f"Schnorr: Server sends: {len(cost.encode('utf-8'))}")

        status = response.get("status")
        decrypted = b'0'
        if status == "no":
            self.console.print("[bold red]thou shall not pass[/bold red]")
            return decrypted

        # print(status)
        assert status == "yes"

        m = response.get("m")# type: ignore
        assert m != None
        return m.encode("utf-8")

    def enroll(self)->bool:
        session_secret = self.perform_schnorr() # returns bytes
        # print(f"Session secret received {session_secret} with type {type(session_secret)}")
        assert session_secret != b'0'
        hasher = HMAC.new(self.shared_secret, digestmod=SHA256) #type: ignore
        assert self.user_id != None
        hasher.update(session_secret + self.user_id.encode("utf-8"))

        encrypted_embeds = []
        embeds = self.embeds[0::2]

        for embed in embeds:
            list1 = [np.round( e * (self.bw**2-1)).astype(np.uint16)
                             for e in embed]
            enc,_ = self.client.encrypt(list1,list1) 
            a = base64.b64encode(enc.serialize()).decode()
            encrypted_embeds.append(a)

        self.client.keys.generate() 
        eval_keys = base64.b64encode(self.client.evaluation_keys.serialize()).decode()
        request = {
            "action": "client_enroll",
            "data": encrypted_embeds,
            "eval_keys": eval_keys, 
            "claimed_id": self.user_id,
            "confirmation": hasher.hexdigest()
        }

        cost = json.dumps(request)
        print(f"Enroll: Client sends: {len(cost.encode('utf-8'))}")

        response = self.send_request(request,2)
        cost = json.dumps(response)
        print(f"Enroll: Server sends: {len(cost.encode('utf-8'))}")
        status = str(response.get("status"))
        if "Failed" in status:
            return False
        else:
            self.console.print("Enrolled")
            return True 
    def authenticate(self, threshold=122) -> bool:
        session_secret = self.perform_schnorr()
        # print(f"Session secret received {session_secret} with type {type(session_secret)}")
        assert session_secret != b'0'
        hasher = HMAC.new(self.shared_secret, digestmod=SHA256) #type: ignore
        assert self.user_id != None
        hasher.update(session_secret + self.user_id.encode("utf-8"))
        encrypted_embeds = []
        embeds = self.embeds[1::2]

        for embed in embeds:
            list1 = [np.round( e * (self.bw**2-1)).astype(np.uint16)
                             for e in embed]
            enc,_ = self.client.encrypt(list1,list1) 
            a = base64.b64encode(enc.serialize()).decode()
            encrypted_embeds.append(a)

        request = {
            "action": "client_authenticate",
            "data": encrypted_embeds, "claimed_id": self.user_id,
            "confirmation": hasher.hexdigest()
        }

        cost = json.dumps(request)
        print(f"Authentication: Client sends: {len(cost.encode('utf-8'))}")
        response = self.send_request(request,3)
        cost = json.dumps(response)
        print(f"Authentication: Server sends: {len(cost.encode('utf-8'))}")
        status = str(response.get("status"))
        if "Failed" in status:
            self.console.print(f"[bold red]Authentication failed {response} with status: {status}[/bold red]")
            return False
        results = response.get("results")
        decisions = []
        for res in results: # type: ignore
            decoded = base64.b64decode(res)
            values = fhe.Value.deserialize(decoded)
            dec = self.client.decrypt(values)
            decisions.append(dec < threshold)

        counts = Counter(decisions)
        if counts[True] > counts[False]:
            self.console.print("Authenticated")
            return True
        else:
            return False

def main():
    id = 2424
    client = Client(id,"./data/embeddings.npz",44,10)

    while True:
        print()
        client.console.print(f"[bold cyan]Client with database id {id}[/bold cyan]")
        client.console.print( "[bold cyan]1. Introduce client[/bold cyan]")
        client.console.print( "[bold cyan]2. Enroll client[/bold cyan]")
        client.console.print( "[bold cyan]3. Authenticate client[/bold cyan]")
        client.console.print( "[bold cyan]4. Exit[/bold cyan]")

        choice = client.console.input("[bold green]Enter your choice (1-4): [/bold green]")

        if choice == '1':
            client.introduce()
        elif choice == '2':
            client.enroll()
        elif choice == '3':
            client.authenticate()
        elif choice == '4':
            client.close()
            client.console.print("[bold red]Exiting...[/bold red]")
            break

        else:
            client.console.print(f"[bold red]Invalid choice {choice}[/bold red]")

if __name__ == "__main__":
    main()
