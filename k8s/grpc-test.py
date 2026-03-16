import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import grpc

print("Testing via port-forward (localhost:8001)...")
try:
    # Create insecure channel for port-forward
    client = grpcclient.InferenceServerClient(
        url="localhost:8001",
        ssl=False,
        verbose=False
    )
    print(f"✓ Server live: {client.is_server_live()}")
    print(f"✓ Server ready: {client.is_server_ready()}")
    metadata = client.get_server_metadata()
    print(f"✓ Server name: {metadata.name}")
    print(f"✓ Server version: {metadata.version}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\nTesting via Gateway (sam-service-triton.int.janelia.org:443)...")
try:
    client = grpcclient.InferenceServerClient(
        url="sam-service-triton.int.janelia.org:443",
        ssl=True,
        verbose=False
    )
    print(f"✓ Server live: {client.is_server_live()}")
    print(f"✓ Server ready: {client.is_server_ready()}")
    metadata = client.get_server_metadata()
    print(f"✓ Server name: {metadata.name}")
    print(f"✓ Server version: {metadata.version}")
except Exception as e:
    print(f"✗ Error: {e}")
