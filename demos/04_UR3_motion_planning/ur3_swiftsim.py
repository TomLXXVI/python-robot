from python_robot.visualisation import SwiftSimulator

from ur3_setup import joint_trajectory

swift_sim = SwiftSimulator()
swift_sim.play_joint_scheme(
    joint_trajectory,
    step=5,
    speed=1.0,
    controls=True
)
