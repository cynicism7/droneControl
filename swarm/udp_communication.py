"""
UDP通信模块
实现无人机之间的UDP通信，用于数据传输和时序同步
"""

import socket
import json
import time
import threading
import numpy as np
from typing import Dict, Callable, Optional

class UDPCommunication:
    """UDP通信类"""
    
    def __init__(self, drone_name, port, leader_port=None):
        """
        初始化UDP通信
        :param drone_name: 无人机名称
        :param port: 本地端口
        :param leader_port: 头机端口（如果是跟随者）
        """
        self.drone_name = drone_name
        self.port = port
        self.leader_port = leader_port
        self.socket = None
        self.is_running = False
        self.received_data = []
        self.message_handlers = {}
        self.sync_timestamp = None
        
    def start(self):
        """启动UDP服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('localhost', self.port))
            self.socket.settimeout(1.0)  # 设置超时，便于检查停止标志
            self.is_running = True
            
            # 启动接收线程
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            
            print(f"[UDP] {self.drone_name} UDP服务器启动在端口 {self.port}")
            return True
        except Exception as e:
            print(f"[UDP错误] {self.drone_name} 启动UDP服务器失败: {e}")
            return False
    
    def stop(self):
        """停止UDP服务器"""
        self.is_running = False
        if self.socket:
            self.socket.close()
        print(f"[UDP] {self.drone_name} UDP服务器已停止")
    
    def _receive_loop(self):
        """接收循环"""
        while self.is_running:
            try:
                data, addr = self.socket.recvfrom(65507)  # UDP最大数据包大小
                message = json.loads(data.decode('utf-8'))
                self._handle_message(message, addr)
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"[UDP错误] {self.drone_name} 接收数据失败: {e}")
    
    def _handle_message(self, message, addr):
        """处理接收到的消息"""
        msg_type = message.get('type')
        
        if msg_type == 'sync':
            # 时序同步消息
            self.sync_timestamp = message.get('timestamp', time.time())
            print(f"[UDP同步] {self.drone_name} 收到同步信号，时间戳: {self.sync_timestamp}")
        elif msg_type == 'data':
            # 数据消息
            self.received_data.append({
                'sender': message.get('sender'),
                'data': message.get('data'),
                'timestamp': message.get('timestamp'),
                'addr': addr
            })
            print(f"[UDP数据] {self.drone_name} 收到来自 {message.get('sender')} 的数据")
        
        # 调用注册的处理器
        if msg_type in self.message_handlers:
            self.message_handlers[msg_type](message, addr)
    
    def send_to_leader(self, data, data_type='data'):
        """
        发送数据到头机
        :param data: 要发送的数据
        :param data_type: 消息类型
        """
        if not self.leader_port:
            return False
        
        try:
            # 对于大数据（如图像、点云），只发送元数据，不发送实际数据
            # 实际数据通过文件系统共享
            if isinstance(data, dict):
                # 提取元数据，移除大型数据
                metadata = {}
                for key, value in data.items():
                    if key == 'image':
                        # 图像数据太大，只发送尺寸信息
                        if hasattr(value, 'shape'):
                            metadata[key] = {
                                'shape': list(value.shape),
                                'dtype': str(value.dtype),
                                'filename': data.get('filename', '')
                            }
                        else:
                            metadata[key] = None
                    elif key == 'points':
                        # 点云数据太大，只发送点数和文件名
                        if isinstance(value, list) and len(value) > 0:
                            metadata[key] = {
                                'point_count': len(value),
                                'filename': data.get('filename', '')
                            }
                        else:
                            metadata[key] = None
                    else:
                        # 其他数据直接复制（位置、姿态等）
                        # 确保numpy类型转换为Python原生类型
                        if isinstance(value, np.ndarray):
                            metadata[key] = value.tolist()
                        elif hasattr(value, 'tolist'):
                            metadata[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            metadata[key] = float(value) if isinstance(value, np.floating) else int(value)
                        else:
                            metadata[key] = value
                
                message = {
                    'type': data_type,
                    'sender': self.drone_name,
                    'data': metadata,
                    'timestamp': time.time()
                }
            else:
                message = {
                    'type': data_type,
                    'sender': self.drone_name,
                    'data': data,
                    'timestamp': time.time()
                }
            
            # 自定义JSON序列化器，处理特殊类型
            def json_serializer(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            data_bytes = json.dumps(message, default=json_serializer).encode('utf-8')
            
            # 检查数据包大小（UDP最大约65507字节，但实际建议小于1500字节）
            MAX_UDP_SIZE = 1400  # 安全大小
            if len(data_bytes) > MAX_UDP_SIZE:
                print(f"[UDP警告] {self.drone_name} 数据包过大 ({len(data_bytes)} 字节)，已压缩为元数据")
            
            self.socket.sendto(data_bytes, ('localhost', self.leader_port))
            return True
        except (TypeError, ValueError) as e:
            print(f"[UDP错误] {self.drone_name} JSON序列化失败: {e}")
            return False
        except OSError as e:
            if e.winerror == 10040:  # Windows错误：消息过大
                print(f"[UDP错误] {self.drone_name} 数据包过大，已自动压缩为元数据")
                # 尝试发送更小的元数据
                try:
                    minimal_metadata = {
                        'type': data_type,
                        'sender': self.drone_name,
                        'filename': data.get('filename', '') if isinstance(data, dict) else '',
                        'timestamp': time.time()
                    }
                    minimal_bytes = json.dumps(minimal_metadata).encode('utf-8')
                    self.socket.sendto(minimal_bytes, ('localhost', self.leader_port))
                    return True
                except:
                    return False
            else:
                print(f"[UDP错误] {self.drone_name} 发送数据失败: {e}")
                return False
        except Exception as e:
            print(f"[UDP错误] {self.drone_name} 发送数据失败: {e}")
            return False
    
    def broadcast_sync(self, timestamp=None):
        """
        广播同步信号（头机使用）
        :param timestamp: 同步时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        
        try:
            message = {
                'type': 'sync',
                'sender': self.drone_name,
                'timestamp': timestamp
            }
            
            data_bytes = json.dumps(message).encode('utf-8')
            # 广播到所有可能的跟随者端口（简化实现，实际应该维护跟随者列表）
            for port in [9002, 9003]:  # 假设跟随者端口
                if port != self.port:
                    try:
                        self.socket.sendto(data_bytes, ('localhost', port))
                    except:
                        pass
            return True
        except Exception as e:
            print(f"[UDP错误] {self.drone_name} 广播同步信号失败: {e}")
            return False
    
    def register_handler(self, msg_type: str, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[msg_type] = handler
    
    def get_sync_timestamp(self):
        """获取同步时间戳"""
        return self.sync_timestamp
    
    def get_received_data(self):
        """获取接收到的数据"""
        return self.received_data
