import socket

# @ref: https://www.cnblogs.com/z-x-y/p/9529930.html
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

print(get_host_ip())
