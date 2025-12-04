#!/usr/bin/env python3
"""
Stress Test for Speculative Request Handling

Simulates interactive segmentation where users:
1. Move mouse rapidly (generating many requests)
2. Cancel previous requests when mouse moves
3. Only care about the final result

This tests:
- Request queuing
- Request cancellation
- Performance under load
"""

import asyncio
import time
import numpy as np
import cv2
from pathlib import Path

from triton_sam import (
    SpeculativeSAM2Client,
    queue_multiple_requests,
    wait_for_latest_result
)


def generate_mouse_path(start, end, num_points=20):
    """
    Generate a smooth path simulating mouse movement.

    Args:
        start: (x, y) starting position
        end: (x, y) ending position
        num_points: Number of intermediate points

    Returns:
        List of (x, y) coordinates
    """
    path = []
    for i in range(num_points):
        t = i / (num_points - 1)
        # Add some randomness to simulate human movement
        jitter_x = np.random.randint(-5, 6) if i > 0 and i < num_points - 1 else 0
        jitter_y = np.random.randint(-5, 6) if i > 0 and i < num_points - 1 else 0

        x = int(start[0] + (end[0] - start[0]) * t + jitter_x)
        y = int(start[1] + (end[1] - start[1]) * t + jitter_y)
        path.append((x, y))

    return path


async def test_speculative_workflow(client, image_path, output_dir):
    """
    Test 1: Simulated interactive workflow with cancellation.

    Simulates a user clicking, moving mouse, and settling on final position.
    """
    print("\n" + "="*70)
    print("Test 1: Interactive Workflow Simulation")
    print("="*70)

    # Encode image once
    print("\n[1/4] Encoding image...")
    start_time = time.time()
    client.set_image(str(image_path))
    encode_time = time.time() - start_time
    print(f"  ✓ Encoded in {encode_time:.3f}s")

    # Define mouse movement: start at one position, move to another
    print("\n[2/4] Simulating mouse movement...")
    start_pos = (100, 100)  # User starts here
    end_pos = (375, 175)    # User ends here (blue rectangle center)
    mouse_path = generate_mouse_path(start_pos, end_pos, num_points=50)

    print(f"  Generated {len(mouse_path)} mouse positions")
    print(f"  Start: {start_pos}")
    print(f"  End: {end_pos}")

    # Queue all requests (simulating rapid mouse movement)
    print("\n[3/4] Queueing speculative requests...")
    session_id = "interactive_session_1"

    coords_list = [np.array([[x, y]]) for x, y in mouse_path]
    labels_list = [np.array([1]) for _ in mouse_path]  # All foreground

    queue_start = time.time()
    tasks = await queue_multiple_requests(
        client, coords_list, labels_list, session_id
    )
    queue_time = time.time() - queue_start

    print(f"  ✓ Queued {len(tasks)} requests in {queue_time:.3f}s")
    print(f"  Average queue time per request: {queue_time/len(tasks)*1000:.2f}ms")

    # Simulate user settling - cancel all but the last few
    await asyncio.sleep(0.1)  # Small delay for realism

    print("\n[4/4] Cancelling intermediate requests...")
    cancelled = client.cancel_session_requests(session_id)
    print(f"  ✓ Marked {cancelled} requests as cancelled")

    # Wait for the final result
    print("\n[5/5] Waiting for final result...")
    result_start = time.time()
    result = await wait_for_latest_result(tasks, client, session_id)
    result_time = time.time() - result_start

    if result is not None:
        masks, iou = result
        print(f"  ✓ Received final mask in {result_time:.3f}s")
        print(f"  IoU confidence: {iou[0, 0]:.3f}")

        # Save result
        mask_logits = masks[0, 0]
        image = cv2.imread(str(image_path))
        mask_resized = cv2.resize(mask_logits, (image.shape[1], image.shape[0]))
        binary = (mask_resized > 0).astype(np.uint8) * 255

        output_path = output_dir / "test1_final_mask.png"
        cv2.imwrite(str(output_path), binary)
        print(f"  Saved mask to {output_path}")
    else:
        print("  ✗ Final request was cancelled or failed")

    # Print statistics
    stats = client.get_session_status(session_id)
    print("\n" + "-"*70)
    print("Session Statistics:")
    for status, count in stats.items():
        print(f"  {status:12s}: {count:3d}")
    print("-"*70)

    client.cleanup_session(session_id)


async def test_burst_cancellation(client, image_path):
    """
    Test 2: Rapid burst of requests with immediate cancellation.

    Tests how the system handles many requests being cancelled quickly.
    """
    print("\n" + "="*70)
    print("Test 2: Burst Cancellation Stress Test")
    print("="*70)

    # Image already encoded from previous test
    session_id = "burst_session_1"

    print("\n[1/3] Generating 100 random requests...")
    num_requests = 100
    coords_list = []
    labels_list = []

    for _ in range(num_requests):
        x = np.random.randint(50, 450)
        y = np.random.randint(50, 450)
        coords_list.append(np.array([[x, y]]))
        labels_list.append(np.array([1]))

    print(f"  Generated {num_requests} random positions")

    print("\n[2/3] Queueing all requests...")
    start_time = time.time()
    tasks = await queue_multiple_requests(
        client, coords_list, labels_list, session_id
    )
    queue_time = time.time() - start_time
    print(f"  ✓ Queued {len(tasks)} requests in {queue_time:.3f}s")

    print("\n[3/3] Immediately cancelling all requests...")
    cancel_start = time.time()
    cancelled = client.cancel_session_requests(session_id)
    cancel_time = time.time() - cancel_start
    print(f"  ✓ Cancelled {cancelled} requests in {cancel_time*1000:.2f}ms")

    # Wait a bit for any in-flight requests to complete
    await asyncio.sleep(1.0)

    # Print statistics
    stats = client.get_session_status(session_id)
    print("\n" + "-"*70)
    print("Session Statistics:")
    for status, count in stats.items():
        print(f"  {status:12s}: {count:3d}")
    print("-"*70)

    client.cleanup_session(session_id)


async def test_multiple_sessions(client, image_path):
    """
    Test 3: Multiple concurrent sessions.

    Tests that session isolation works correctly.
    """
    print("\n" + "="*70)
    print("Test 3: Multiple Concurrent Sessions")
    print("="*70)

    num_sessions = 5
    requests_per_session = 20

    print(f"\n[1/3] Creating {num_sessions} concurrent sessions...")
    print(f"  Each session will have {requests_per_session} requests")

    all_tasks = []
    session_ids = []

    start_time = time.time()
    for i in range(num_sessions):
        session_id = f"session_{i}"
        session_ids.append(session_id)

        # Random positions for this session
        coords_list = [
            np.array([[np.random.randint(50, 450), np.random.randint(50, 450)]])
            for _ in range(requests_per_session)
        ]
        labels_list = [np.array([1]) for _ in range(requests_per_session)]

        tasks = await queue_multiple_requests(
            client, coords_list, labels_list, session_id
        )
        all_tasks.append((session_id, tasks))

    queue_time = time.time() - start_time
    total_requests = num_sessions * requests_per_session
    print(f"  ✓ Queued {total_requests} requests across {num_sessions} sessions")
    print(f"  Total time: {queue_time:.3f}s")

    print("\n[2/3] Cancelling specific sessions...")
    # Cancel odd-numbered sessions
    for i, session_id in enumerate(session_ids):
        if i % 2 == 1:
            cancelled = client.cancel_session_requests(session_id)
            print(f"  Session {i}: Cancelled {cancelled} requests")

    print("\n[3/3] Waiting for even-numbered sessions to complete...")
    # Wait for even sessions (not cancelled)
    results_received = 0
    for i, (session_id, tasks) in enumerate(all_tasks):
        if i % 2 == 0:  # Even sessions
            result = await wait_for_latest_result(tasks, client, session_id)
            if result is not None:
                results_received += 1

    print(f"  ✓ Received {results_received} final results")

    # Print statistics for each session
    print("\n" + "-"*70)
    print("Per-Session Statistics:")
    for session_id in session_ids:
        stats = client.get_session_status(session_id)
        status_str = ", ".join([f"{k}={v}" for k, v in stats.items()])
        print(f"  {session_id:15s}: {status_str}")
    print("-"*70)

    # Cleanup
    for session_id in session_ids:
        client.cleanup_session(session_id)


async def main():
    """Run all stress tests."""
    # Setup
    test_dir = Path("test")
    image_dir = test_dir / "images"
    output_dir = test_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "test_shapes.jpg"
    if not image_path.exists():
        print(f"Error: Test image not found at {image_path}")
        print("Run 'pixi run test-sam2' first to generate test images")
        return

    print("="*70)
    print("SAM2 Speculative Request Stress Test")
    print("="*70)
    print(f"\nTest image: {image_path}")
    print(f"Output directory: {output_dir}")

    # Initialize client
    print("\nInitializing client...")
    client = SpeculativeSAM2Client(url="localhost:8000")

    try:
        # Run tests
        await test_speculative_workflow(client, image_path, output_dir)
        await test_burst_cancellation(client, image_path)
        await test_multiple_sessions(client, image_path)

        print("\n" + "="*70)
        print("All Tests Complete!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
