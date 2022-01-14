class GrandpaA:
    grandpa_attr = 50
    @staticmethod
    def grandpa_a_fun():
        print("this is grandpa_a")


class GrandpaB:
    @staticmethod
    def grandpa_b_fun():
        print("this is grandpa_b")


"""
方案一：
"""
def modify_base_class(base_name):
    class RosbridgeWebSocket(base_name):
        client_id_seed = 0
        clients_connected = 0
        fragment_timeout = 600  # seconds
        # protocol.py:
        delay_between_messages = 0  # seconds
        max_message_size = None  # bytes
        unregister_timeout = 10.0  # seconds
        bson_only_mode = False

        # 连接成功时调用
        def onOpen(self):
            cls = self.__class__
            parameters = {
                "fragment_timeout": cls.fragment_timeout,
                "delay_between_messages": cls.delay_between_messages,
                "max_message_size": cls.max_message_size,
                "unregister_timeout": cls.unregister_timeout,
                "bson_only_mode": cls.bson_only_mode
            }
            try:
                # 创建客户端传输协议
                self.protocol = RosbridgeProtocol(9999, self.shared_data, parameters=parameters)
                # 创建消息队列
                self.incoming_queue = IncomingQueue(self.protocol)
                # 启动队列线程
                self.incoming_queue.start()
                # 创建用于暂停传输的对象
                producer = OutgoingValve(self)
                self.transport.registerProducer(producer, True)
                producer.resumeProducing()
                self.protocol.outgoing = producer.relay
                # 生成客户端随机id
                cls.client_id_seed += 1
                cls.clients_connected += 1
                self.client_id = uuid.uuid4()
                self.peer = self.transport.getPeer().host
                self.shared_data['client_connected'] = True
            except Exception as exc:
                rospy.logerr("Unable to accept incoming connection.  Reason: %s", str(exc))
            rospy.loginfo("Client connected.  %d clients total.", cls.clients_connected)

        # 接收到消息时调用
        def onMessage(self, message, binary):
            message_str = str(message)
            if 'type' in message_str and 'subscribe' in message_str or 'advertise' in message_str:
                print('*********', message_str)
            if not binary:
                message = message.decode('utf-8')
            # print(type(json.loads(str(message))))
            self.incoming_queue.push(message)

        def outgoing(self, message):
            if type(message) == bytearray:
                binary = True
                message = bytes(message)
            else:
                binary = False
                message = message.encode('utf-8')
            self.sendMessage(message, binary)

        # 链接断开时调用
        def onClose(self, was_clean, code, reason):
            if not hasattr(self, 'protocol'):
                return
            cls = self.__class__
            cls.clients_connected -= 1
            rospy.loginfo("Client disconnected. %d clients total.", cls.clients_connected)
            self.shared_data['client_connected'] = False
            self.incoming_queue.finish()

    return Father


"""
方案二
"""
# class Father(GrandpaA):
#     @staticmethod
#     def father_fun():
#         pass
#
# def modify_base_class(base_name):
#     Father.__bases__ = base_name,
#     return Father


if __name__ == '__main__':
    father_a = modify_base_class(GrandpaA)
    father_a.grandpa_a_fun()
    father_b = modify_base_class(GrandpaB)
    father_b.grandpa_b_fun()
    print(father_a.__bases__)  # (<class '__main__.GrandpaA'>,) return: tuple
    print(father_b.__bases__)  # (<class '__main__.GrandpaB'>,) return: tuple

