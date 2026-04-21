import socket
import json
import argparse
import base64
import numpy as np
import cv2
import time

def create_tcp_connection(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    return sock


def send_json(sock, data):
    json_data = json.dumps(data, ensure_ascii=False) + '\n'
    sock.sendall(json_data.encode('utf-8'))


recv_buffer = ""

def receive_response(sock):
    global recv_buffer
    while '\n' not in recv_buffer:
        part = sock.recv(4096).decode('utf-8')
        if not part:
            break
        recv_buffer += part
    if '\n' in recv_buffer:
        line, recv_buffer = recv_buffer.split('\n', 1)
        return line.strip()
    else:
        line, recv_buffer = recv_buffer, ""
        return line.strip()

def close_connection(sock):
    if sock:
        sock.close()


def create_init_data(response_format, deivce, enoutput, frame_height, frame_width, enable_webstream, rtsp):
    return {
        "request_id": "camera_001",
        "work_id": "camera",
        "action": "setup",
        "object": "camera.setup",
        "data": {
            "response_format": "image.yuvraw.base64" if response_format =="yuv" else "image.jpeg.base64",
            "input": deivce,
            "enoutput": enoutput,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "enable_webstream": enable_webstream,
            "VinParam.bAiispEnable": 0,
            "rtsp": "rtsp.1280x720.h265" if rtsp == "h265" else "rtsp.1280x720.h264",
        }
    }


def parse_setup_response(response_data, sent_request_id):
    error = response_data.get('error')
    request_id = response_data.get('request_id')

    if request_id != sent_request_id:
        print(f"Request ID mismatch: sent {sent_request_id}, received {request_id}")
        return None

    if error and error.get('code') != 0:
        print(f"Error Code: {error['code']}, Message: {error['message']}")
        return None

    return response_data.get('work_id')


def reset(sock):
    sent_request_id = 'reset_000'
    reset_data = {
        "request_id": sent_request_id,
        "work_id": "sys",
        "action": "reset"
    }
    ping_data = {
        "request_id": "ping_000",
        "work_id": "sys",
        "action": "ping"
    }
    send_json(sock, reset_data)
    while True:
        try:
            send_json(sock, ping_data)
            time.sleep(1)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            return


def setup(sock, init_data):
    sent_request_id = init_data['request_id']
    send_json(sock, init_data)
    response = receive_response(sock)
    response_data = json.loads(response)
    return parse_setup_response(response_data, sent_request_id)


def exit_session(sock, deinit_data):
    send_json(sock, deinit_data)
    print("Exit")


def parse_inference_response(response_data):
    error = response_data.get('error')
    if error and error.get('code') != 0:
        print(f"Error Code: {error['code']}, Message: {error['message']}")
        return None

    return response_data.get('data')


def main(args):
    sock = create_tcp_connection(args.host, args.port)

    frame_width, frame_height = args.imgsz

    try:
        print("Reset...")
        reset(sock)
        close_connection(sock)
        sock = create_tcp_connection(args.host, args.port)

        print("Setup Camera...")
        init_data = create_init_data(
            response_format = args.format,
            enoutput=args.enoutput,
            deivce=args.device,
            frame_height=frame_height,
            frame_width=frame_width,
            enable_webstream=args.webstream,
            rtsp=args.rtsp
        )
        camera_work_id = setup(sock, init_data)
        print("Setup Camera finished.")

        while True:
            response = receive_response(sock)
            if not response:
                continue
            response_data = json.loads(response)

            data = parse_inference_response(response_data)
            if data is None:
                break

            decoded = base64.b64decode(data)
            if args.format == "yuv":
                yuv_frame = np.frombuffer(decoded, dtype=np.uint8).reshape((frame_height, frame_width, 2))
                bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_YUY2)
            else:
                jpg_array = np.frombuffer(decoded, dtype=np.uint8)
                bgr_frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)

            cv2.imshow("Camera Frame", bgr_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        exit_session(sock, {
            "request_id": "camera_exit",
            "work_id": camera_work_id,
            "action": "exit"
        })
        time.sleep(3) # Allow time for the exit command to be processed
    finally:
        close_connection(sock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCP Client to send JSON data.")
    parser.add_argument("--host", type=str, default="localhost", help="Server hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=10001, help="Server port (default: 10001)")
    parser.add_argument("--device", type=str, default="axera_single_sc850sl", help="Camera name, i.e. axera_single_sc850sl or /dev/video0")
    parser.add_argument("--enoutput", type=bool, default=True, help="Whether to output image data")
    parser.add_argument("--format", "--output-format", type=str, default="jpeg", help="Output image data format, i.e. jpeg or yuv")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[320, 320], help="image (h, w)")
    parser.add_argument("--webstream", action="store_true", help="Enable webstream")
    parser.add_argument("--rtsp", default="h264", help="rtsp output, i.e. h264 or h265")


    args = parser.parse_args()
    main(args) 