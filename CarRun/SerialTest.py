'''
sudo chmod 777 /dev/ttyACM0
'''

import time

import serial #导入serial模块
port = "/dev/ttyACM0" #Arduino端口
ser = serial.Serial(port,9600,timeout=0.001) #设置端口，每秒回复一个信息
ser.flushInput() #清空缓冲器

while True:
	ser.write(b'1') #将'1'字符转换为字节发送
	response = ser.read()	#.decode('utf-8')将数据转换成str格式
	print(response)

	time.sleep(0.05)
		

