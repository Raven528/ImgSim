import asyncio
import websockets
import json

# WebSocket 服务器的地址
SERVER_URI = "ws://localhost:8765"

# 客户端主函数
async def client():
    """
    客户端连接到 WebSocket 服务器。
    - 向服务器发送数据更新请求。
    - 接收来自服务器的实时更新消息。
    """
    async with websockets.connect(SERVER_URI) as websocket:
        print(f"Connected to server: {SERVER_URI}")

        # 启动发送任务和接收任务
        send_task = asyncio.create_task(send_messages(websocket))
        receive_task = asyncio.create_task(receive_messages(websocket))

        # 等待任务完成
        await asyncio.wait([send_task, receive_task], return_when=asyncio.FIRST_COMPLETED)

# 发送消息的任务
async def send_messages(websocket):
    """
    向服务器发送更新共享数据的消息。
    """
    try:
        while True:
            # 从用户输入读取键值对
            key = input("Enter key to update (or 'exit' to quit): ").strip()
            if key.lower() == "exit":
                print("Closing connection...")
                break

            value = input(f"Enter value for key '{key}': ").strip()

            # 构造 JSON 消息
            message = json.dumps({"action": "update", "data": {key: value}})
            # 发送消息到服务器
            await websocket.send(message)
            print(f"Sent: {message}")
    except Exception as e:
        print(f"Error while sending messages: {e}")

# 接收消息的任务
async def receive_messages(websocket):
    """
    接收来自服务器的消息并显示。
    """
    try:
        while True:
            # 等待接收服务器的消息
            message = await websocket.recv()
            # 将 JSON 消息解析为字典
            data = json.loads(message)
            print(f"Received from server: {data}")
    except websockets.ConnectionClosed:
        print("Connection closed by server.")
    except Exception as e:
        print(f"Error while receiving messages: {e}")

# 脚本入口
if __name__ == "__main__":
    # 启动客户端
    asyncio.run(client())
