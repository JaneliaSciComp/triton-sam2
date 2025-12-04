"""
Triton SAM2 - Production deployment of SAM2 on NVIDIA Triton Inference Server

This package provides Python clients for interacting with SAM2 models deployed
on Triton Inference Server, supporting both basic and speculative request workflows.
"""

from triton_sam2.client import SAM2TritonClient
from triton_sam2.speculative_client import (
    SpeculativeSAM2Client,
    queue_multiple_requests,
    wait_for_latest_result
)

__version__ = "0.1.0"

__all__ = [
    "SAM2TritonClient",
    "SpeculativeSAM2Client",
    "queue_multiple_requests",
    "wait_for_latest_result",
]
