"""Test WebSocket connection."""
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws/stream/test-session"
    
    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✓ Connected!")
            
            # Send start_session message
            print("Sending start_session...")
            await websocket.send(json.dumps({"type": "start_session"}))
            print("✓ Message sent")
            
            # Wait for responses
            print("Waiting for messages...")
            for i in range(10):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    print(f"← Received: {data.get('type')}")
                except asyncio.TimeoutError:
                    print(f"  (timeout {i+1}/10)")
                    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket())
