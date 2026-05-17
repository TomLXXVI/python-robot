from python_robot.visualisation import SwiftSimulator

from ur3_setup import joint_scheme

swift_sim = SwiftSimulator()
swift_sim.play_joint_scheme(
    joint_scheme,
    step=5,
    speed=1.0,
    controls=True
)
