from adafruit_servokit import ServoKit

kit = ServoKit(channels = 16)
kit.servo[0].set_pulse_width_range(500, 2500)
kit.servo[1].set_pulse_width_range(500, 2500)
while True:
    angle = int(input("input angle: "))
    kit.servo[0].angle = angle
    kit.servo[1].angle = angle
