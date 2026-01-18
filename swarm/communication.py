"""
无人机通信模块
模拟无人机之间的数据传输，用于将采集的数据传输到头机
"""

class DroneNetwork:
    """无人机网络通信类"""
    
    def __init__(self, leader_drone):
        """
        初始化无人机网络
        :param leader_drone: 头机（接收数据的无人机）
        """
        self.leader = leader_drone
        self.drones = {}  # 存储所有注册的无人机
        self.received_data = []  # 头机接收到的数据
        
    def register_drone(self, drone):
        """注册无人机到网络"""
        self.drones[drone.name] = drone
        
    def send_data_to_leader(self, sender_drone, data):
        """
        将数据发送到头机
        :param sender_drone: 发送数据的无人机
        :param data: 要发送的数据（图像信息）
        :return: 是否发送成功
        """
        if sender_drone.name == self.leader.name:
            # 头机自己的数据直接添加
            self.received_data.append({
                'sender': sender_drone.name,
                'data': data,
                'timestamp': data.get('timestamp', 0)
            })
            return True
        
        # 模拟数据传输（在实际系统中这里会有网络延迟、丢包等）
        # 在仿真环境中，我们直接传输
        try:
            # 确保timestamp是Python原生类型
            timestamp = data.get('timestamp', 0)
            if hasattr(timestamp, '__int__'):
                timestamp = int(timestamp)
            
            self.received_data.append({
                'sender': sender_drone.name,
                'data': data,
                'timestamp': timestamp
            })
            print(f"[通信] {sender_drone.name} 向 {self.leader.name} 发送了图像数据")
            return True
        except Exception as e:
            print(f"[通信错误] {sender_drone.name} 发送数据失败: {e}")
            return False
    
    def send_all_images_to_leader(self, sender_drone):
        """将发送者的所有图像发送到头机"""
        images = sender_drone.get_captured_images()
        success_count = 0
        for img_info in images:
            if self.send_data_to_leader(sender_drone, img_info):
                success_count += 1
        print(f"[通信] {sender_drone.name} 共发送 {success_count}/{len(images)} 张图像到头机")
        return success_count
    
    def get_received_data(self):
        """获取头机接收到的所有数据"""
        return self.received_data
    
    def get_images_by_sender(self, sender_name):
        """按发送者名称获取图像"""
        return [item['data'] for item in self.received_data if item['sender'] == sender_name]
