import asyncio
import websockets
import json
from pathlib import Path
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

DATA_FILE = Path("WebSocket/shared_data.json")

if not DATA_FILE.exists():
    with open(DATA_FILE, "w") as f:
        json.dump({}, f)  # 初始化为空字典

clients = set()

async def handler(websocket):
    """
    每当有新的客户端连接时，调用此函数。
    负责管理连接的生命周期和消息处理。
    """
    clients.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)

            if data.get("action") == "update":
                with open(DATA_FILE, "r+") as f:
                    shared_data = json.load(f)  # 读取当前共享数据
                    shared_data.update(data["data"])  # 更新数据
                    f.seek(0)  # 回到文件开头
                    json.dump(shared_data, f)  # 写入更新后的数据
                    f.truncate()  # 清除多余内容（防止新数据比旧数据短时出现问题）

                # 构造更新消息，通知所有客户端
                update_message = json.dumps({"action": "update", "data": shared_data})
                await asyncio.gather(*[client.send(update_message) for client in clients])

    except websockets.ConnectionClosed:
        print("A client disconnected.")
    finally:
        clients.remove(websocket)

async def main():
    """
    启动 WebSocket 服务器，监听客户端连接。
    """
    async with websockets.serve(handler, "localhost", 8765):  # 监听地址和端口
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

