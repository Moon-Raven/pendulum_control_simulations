import feedback_control as fc
import gain_scheduling as gs

if __name__ == '__main__':
    print('Program start!')

    # The following is a list of simulation examples which can be
    # uncommented and executed.

    # fc.main01_constant_input_simulation()
    # fc.main02_constant_input_simulation_fixed_delta()
    # fc.main03_feedback_state_stabilization()
    # fc.main04_output_feedback_stabilization()
    # fc.main05_integral_control()
    # gs.main01_fixed_alpha()
    # gs.main02_gain_scheduling_unmodified()
    # gs.main03_gain_scheduling_modified()
    # gs.main04_gain_scheduling_modified_ramp()
    gs.main05_gain_scheduling_modified_steep_ramp()

    print('Program end!')
