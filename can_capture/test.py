import serial

ser = serial.Serial('COM5', 1000000)

while True:
    print(ser.readline().decode(errors='ignore').strip())