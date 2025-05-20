import base64
import socket
import json
import uuid
import threading
from crypto.secp256k1 import secp256k1_curve as curve
from random import SystemRandom
from concrete import fhe
import os
from crypto.EClib import Point
from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA512
from rich.progress import Progress
from rich.console import Console
from src.helpers import clear_console
import time
from Crypto.Hash import HMAC, SHA256

class Server:
    def __init__(self, host='localhost', port=8888):
        self.console = Console()
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.running = False

        # Dictionary to hold id <--> pk associations
        self.pks = {}
        # Dictionary to hold id <--> enc_embeds associations
        self.database = {}
        # Dictionary to hold id <--> eval_keys associations
        self.eval_keys = {}
        # Dictionary to hold id <--> [I,c] associations for schnorr protocol
        self.schnorr = {}

        # Dictionary to hold id <--> HMAC hasher associations for session secret checks
        self.hashers = {}

        # Dictionary to hold id <--> shared_secret associations 
        self.shared_secrets = {}
        # Dictionary to hold id <--> session_secret associations 
        self.session_secrets = {}
        # List to keep for used random numbers --> Stateful
        self.used_randomness = []

        self.console.print("[bold yellow]Creating sk,pk for secp256k1...[/bold yellow]")
        self.prg = SystemRandom()
        self.sk = self.prg.randrange(1,curve.generator.n-1)
        self.pk = self.sk * curve.generator.G
        self.console.print("[bold yellow]Loading server file ...[/bold yellow]")
        self.server = fhe.Server.load(os.getcwd() + "/server.zip")
    
    def start(self):
        clear_console()
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.running = True
        self.console.print(f"[bold yellow]Server started on {self.host}:{self.port}[/bold yellow]")
        
        try:
            while self.running:
                client_socket, address = self.socket.accept()
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket, address))
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            self.console.print("[bold red]Server shutting down...[/bold red]")
        finally:
            self.socket.close()

    
    def handle_client(self, client_socket, address):
        self.console.print(f"[bold green]Connection from {address}[/bold green]")
        buffer = ""  # Store incomplete data
    
        try:
            while True:
                data = client_socket.recv(1024 * 600).decode('utf-8')
                if not data:
                    break
                buffer += data
    
                while "\n" in buffer:  # Process complete messages
                    message, buffer = buffer.split("\n", 1)  # Extract one message
                    try:
                        request = json.loads(message.strip())  # Ensure clean JSON parsing
                        response = self.process_request(request)
                        client_socket.send((json.dumps(response) + "\n").encode('utf-8'))
                    except json.JSONDecodeError:
                        client_socket.send(
                            json.dumps( {"status": "error",
                                         "message": "Invalid JSON"}
                                       ).encode('utf-8')
                        )
        except Exception as e:
            self.console.print(f"[bold red]Error handling client: {e} [/bold red]")
        finally:
            client_socket.close()
            self.console.print(f"[bold red]Connection from {address} closed.[/bold red]")

    def process_request(self, request):
        action = request.get('action')
        self.console.print(f"Process request action {action} with type {type(json.dumps(request).encode('utf-8'))} with size: {len(json.dumps(request).encode('utf-8'))}")
        if action == "introduce":
            return self.introduce(request)
        elif action == 'client_hello':
            return self.challenge(request)
        elif action == 'client_response':
            return self.schnorr_finish(request)
        elif action == 'client_enroll':
            return self.client_enroll(request)
        elif action == 'client_authenticate':
            return self.client_authenticate(request)
        else:
            return {"status": "error", "message": "Unknown action"}
    
    def introduce(self, request):
        user_id = str(uuid.uuid4())
        self.pks[user_id] = Point(curve, request.get("pk")[0], request.get("pk")[1]) #type: ignore
        fhe_client_specs = self.server.client_specs.serialize().decode()
        salt = self.prg.getrandbits(128)
        response = {
                "pk": [self.pk.x, self.pk.y],
                "id": user_id,
            "salt": salt,
            "specs": fhe_client_specs 
        }
        shared_secret =  (self.sk * self.pks[user_id]).x.to_bytes(32,'big')  
        self.shared_secrets[user_id] = HKDF(shared_secret, 32, salt.to_bytes(16,'big'), SHA512)
        return response


    def challenge(self, request):
        # Generate randomness
        while True:
            c = self.prg.randint(1,curve.generator.n-1)
            if c not in self.used_randomness:
                self.used_randomness.append(c)
                break
        response = {"c":c}
        self.schnorr[request.get("claimed_id")] = [request,c]
        return response

    def schnorr_finish(self, request):
        s = request.get("s")
        claimed_id = request.get("claimed_id")
        I_coords = self.schnorr[claimed_id][0].get("I")
        I = Point(curve,I_coords[0], I_coords[1])
        c = self.schnorr[claimed_id][1]
        sG = s * curve.generator.G
        rhs = I + c * self.pks[claimed_id]
        if sG == rhs:
            session_secret = self.prg.getrandbits(128).to_bytes(16, 'big')
            self.hashers[claimed_id] = HMAC.new(self.shared_secrets[claimed_id], digestmod=SHA256)
            timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime()) 
            m = session_secret.hex() + timestamp 
            self.session_secrets[claimed_id] = m
            response = {
                "status": "yes",
                "m": m,
            }
            # print(f"Sent message m: {m}")
        else:
            self.session_secrets[claimed_id] = None
            response = {
                "status": "no"
            }
        self.schnorr[claimed_id] = None
        return response

    def client_enroll(self, request):
        claimed_id = request.get("claimed_id")
        for k in self.database.keys():
            if k == claimed_id:
                return {"status": f"Failed: client id {claimed_id} already enrolled"}
        confirmation = request.get("confirmation")

        # print(f"Enroll Confirmation at the server for {claimed_id}{type(claimed_id)}: {confirmation} with type {type(confirmation)}")
        self.hashers[claimed_id].update((self.session_secrets[claimed_id] + claimed_id).encode("utf-8"))
        if confirmation != self.hashers[claimed_id].hexdigest():
            return {"status" : "Failed: hash check of session secret"}
        
        assert confirmation == self.hashers[claimed_id].hexdigest() 
        eval_keys = request.get("eval_keys")
        eval_keys = base64.b64decode(eval_keys)
        eval_keys = fhe.EvaluationKeys.deserialize(eval_keys)
        self.eval_keys[claimed_id] = eval_keys

        data = request.get("data")
        received_embeddings = []
        for embed in data:
            decoded = base64.b64decode(embed)
            values = fhe.Value.deserialize(decoded)
            received_embeddings.append(values)

        self.database[claimed_id] = received_embeddings
        return {"status":"Success: client enrolled"}

    def client_authenticate(self, request):
        claimed_id = request.get("claimed_id")
        if claimed_id not in self.database:
            return {"status": f"Failed: client id {claimed_id} not in database"}
        confirmation = request.get("confirmation")

        # print(f"Authenticate Confirmation at the server for {claimed_id}{type(claimed_id)}: {confirmation} with type {type(confirmation)}")
        self.hashers[claimed_id].update((self.session_secrets[claimed_id] + claimed_id).encode("utf-8"))
        if confirmation != self.hashers[claimed_id].hexdigest():
            return {"status" : "Failed: hash check of session secret"}
        
        assert confirmation == self.hashers[claimed_id].hexdigest() 
        data = request.get("data")
        results = []

        total_iterations = len(data) * len(self.database[claimed_id])
        with Progress() as progress:
            task = progress.add_task("[bold yellow]Computing distance on encrypted data...[/bold yellow]",
                                         total=total_iterations)
            for new_embed in data:
                decoded = base64.b64decode(new_embed)
                values = fhe.Value.deserialize(decoded)
                for stored_embed in self.database[claimed_id]:
                    res = self.server.run(values, stored_embed,
                                        evaluation_keys=self.eval_keys[claimed_id])

                    a = base64.b64encode(res.serialize()).decode()
                    results.append(a)
                    progress.update(task, advance=1)
        response = {
            "status" : "Sucess",
            "results" : results
        }
        return response

        
if __name__ == "__main__":
    server = Server()
    server.start()
