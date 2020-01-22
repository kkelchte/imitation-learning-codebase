
"""Listens to FSM state and changes controls accordingly

Configuration defines which FSM state corresponds to which control connection
Each FSM state has a number of potential actors or controls steering the robot:
Running -> EXPERT / DNN / USER
TakeOff -> EXPERT
TakeOver -> USER
DriveBack -> DB

Config defines for required FSM states the corresponding topics to be connected to the robot cmd_vel.
"""