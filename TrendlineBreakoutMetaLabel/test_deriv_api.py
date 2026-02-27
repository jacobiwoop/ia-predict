import asyncio
import websockets
import json
import time

async def test_api():
    url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    async with websockets.connect(url) as ws:
        now = int(time.time())
        two_days_ago = now - 2 * 24 * 3600
        
        request = {
            "ticks_history": "R_75",
            "adjust_start_time": 1,
            "count": 5,
            "end": str(1754226000),  # 2025-08-03
            "granularity": 3600,
            "style": "candles"
        }
        await ws.send(json.dumps(request))
        res = json.loads(await ws.recv())
        print("Test 5 (end is 2025-08-03):")
        if 'candles' in res:
            for c in res['candles']:
                print(c['epoch'])
        else:
            print(res)

asyncio.run(test_api())
