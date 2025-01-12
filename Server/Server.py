import websockets
import asyncio
message = "/"

####################

##### 서버 비동기함수 #####
async def code(websocket, path):
    
    # 클라이언트 파악
    global message
    name = await asyncio.wait_for(websocket.recv(), timeout=1e5)
    print("From", name, ":", "Connected")
    await websocket.send("Connected")
    
    # 클라이언트가 카메라
    if(name == "Camera"):
        while(True):
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1e5)
            except Exception:
                continue
            if( message != "/" ):
                print("From", name, ":", message)
            await websocket.send("")
    
    # 클라이언트가 젯레이서
    else:
        while(True):
            try:
                await asyncio.wait_for(websocket.recv(), timeout=1e5)
            except Exception:
                continue
            if( len(message) > 3 and message[:2] == name ):
                await websocket.send(message[3:])
                print("To", name, ":", message[3:])
            else:
                await websocket.send("")

####################

##### 서버 열기 ##### 

# 서버의 웹소켓/IP/Port 지정
server = websockets.serve(code, "192.168.0.206", 8000)

# 서버를 무한이벤트루프로 열면, 클라이언트마다 비동기함수가 할당
asyncio.get_event_loop().run_until_complete(server)
asyncio.get_event_loop().run_forever()