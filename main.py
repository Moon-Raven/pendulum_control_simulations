import feedback_control as fc
import gain_scheduling as gs
import tracking as tr
import sliding_mode_control as smc

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
    # gs.main05_gain_scheduling_modified_steep_ramp()
    # tr.main01_reference_model()
    # tr.main02_pendulum_simple()
    # tr.main03_pendulum_simple_initial_cond()
    # tr.main04_pendulum_simple_perturbed()
    # tr.main05_robot1()
    # tr.main06_robot2()
    # smc.main01_check_matrices()
    # smc.main02_simulate_fixed_g()
    # smc.main03_simulate_dynamic_g_small_tau()
    # smc.main04_simulate_dynamic_g_large_tau()

    print('Program end!')
