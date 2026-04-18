import os
import sys

from Task1 import (
    XI_1, XI_2, LAMBDA, GAMMA,
    COST_PREVENTIVE, COST_CORRECTIVE, COST_UNAVAILABILITY,
    TIME_TRAVEL, TIME_PREVENTIVE, TIME_CORRECTIVE,
    create_state_space, create_state_index_map
)
from Task3 import task3_evaluate_failure_only, task3_monte_carlo_verification
from Task4 import task4_value_iteration, plot_convergence

# Get the directory where main.py is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# MAIN FUNCTIONS

def run_all():

    print("ASSIGNMENT 1 - OPTIMAL MAINTENANCE IN A NETWORK")
    print(f"\nMDP Parameters:")
    print(f"  Machine 1 failure threshold (xi_1): {XI_1}")
    print(f"  Machine 2 failure threshold (xi_2): {XI_2}")
    print(f"  Poisson degradation parameter (lambda): {LAMBDA}")
    print(f"  Discount factor (gamma): {GAMMA}")
    print(f"  Preventive maintenance cost: {COST_PREVENTIVE}")
    print(f"  Corrective maintenance cost: {COST_CORRECTIVE}")
    print(f"  Unavailability cost per time unit: {COST_UNAVAILABILITY}")
    print(f"  Travel time: {TIME_TRAVEL} time unit(s)")
    print(f"  Preventive maintenance time: {TIME_PREVENTIVE} time unit(s)")
    print(f"  Corrective maintenance time: {TIME_CORRECTIVE} time unit(s)")

    # Task 1: Show state space info
    states = create_state_space()
    state_index = create_state_index_map(states)
    print(f"\nTotal states: {len(states)}")
    
    # Task 3: Evaluate corrective-only policy
    V_failure, states, state_index, iterations_3 = task3_evaluate_failure_only()
    
    # Task 3 Verification with Monte Carlo (no plot)
    mc_results = task3_monte_carlo_verification(generate_plot=False)
    
    # Task 4: Value Iteration
    V_optimal, pi_optimal, states, state_index, iterations_4, convergence = task4_value_iteration()
    
    # Compare policies
    print("\nCOMPARISON: CORRECTIVE-ONLY vs OPTIMAL POLICY")
    initial_idx = state_index[(0, 0, 0, 0, 0)]
    
    cost_failure = V_failure[initial_idx]
    cost_optimal = V_optimal[initial_idx]
    improvement = cost_failure - cost_optimal
    improvement_pct = (improvement / cost_failure) * 100
    
    print(f"Corrective-only policy: {cost_failure:.6f}")
    print(f"Optimal policy:         {cost_optimal:.6f}")
    print(f"Improvement:            {improvement:.6f} ({improvement_pct:.2f}%)")
    
    print("\nSUMMARY")
    print(f"Task 3: Expected cost under corrective-only = {cost_failure:.6f}")
    print(f"        (converged in {iterations_3} iterations)")
    print(f"\nTask 4: Expected cost under optimal policy = {cost_optimal:.6f}")
    print(f"        (converged in {iterations_4} iterations)")
    print(f"\nOptimal policy achieves {improvement_pct:.2f}% cost reduction")
    
    return {
        'V_failure': V_failure,
        'V_optimal': V_optimal,
        'pi_optimal': pi_optimal,
        'states': states,
        'state_index': state_index,
        'iterations_3': iterations_3,
        'iterations_4': iterations_4,
        'convergence': convergence,
        'cost_failure': cost_failure,
        'cost_optimal': cost_optimal,
        'improvement_pct': improvement_pct
    }


def generate_results_txt(output_path=None):
    # Generate results.txt file with all output.

    if output_path is None:
        output_path = os.path.join(SCRIPT_DIR, 'results.txt')
    
    # Redirect stdout to file
    original_stdout = sys.stdout
    with open(output_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        run_all()
        sys.stdout = original_stdout
    
    print(f"Results saved to: {output_path}")


def generate_vi_convergence_plot(output_path=None):

    # Generate Value Iteration convergence plot (value_iteration_convergence.png).
    
    if output_path is None:
        output_path = os.path.join(SCRIPT_DIR, 'value_iteration_convergence.png')
    
    # Run value iteration to get convergence data
    print("Running Value Iteration...")
    states = create_state_space()
    state_index = create_state_index_map(states)
    _, _, _, _, _, convergence = task4_value_iteration()
    
    # Generate plot
    plot_convergence(convergence, save_path=output_path)


def generate_mc_convergence_plot(output_path=None):

    # Generate Monte Carlo convergence plot (monte_carlo_convergence.png).

    if output_path is None:
        output_path = os.path.join(SCRIPT_DIR, 'monte_carlo_convergence.png')
    
    # Generate MC plot
    print("Running Monte Carlo simulations for convergence plot...")
    task3_monte_carlo_verification(generate_plot=True, plot_path=output_path)


def generate_all_files():
    # Generate all output files: results.txt and both PNG plots
    generate_results_txt()
    generate_vi_convergence_plot()
    generate_mc_convergence_plot()
    print("\nAll files generated!")


# MAIN EXECUTION

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Assignment 1 2026 - Optimal Maintenance')
    parser.add_argument('--generate-all', action='store_true', help='Generate all output files')
    parser.add_argument('--generate-results', action='store_true', help='Generate results.txt')
    parser.add_argument('--generate-vi-plot', action='store_true', help='Generate VI convergence plot')
    parser.add_argument('--generate-mc-plot', action='store_true', help='Generate MC convergence plot')
    args = parser.parse_args()
    
    if args.generate_all:
        generate_all_files()
    elif args.generate_results:
        generate_results_txt()
    elif args.generate_vi_plot:
        generate_vi_convergence_plot()
    elif args.generate_mc_plot:
        generate_mc_convergence_plot()
    else:
        # Default: just run all tasks (no file generation)
        run_all()


# TERMINAL COMMANDS (run from the Solution directory)

# 1. Generate output (run all tasks, print to console):
#    python main.py
#
# 2. Generate results.txt:
#    python main.py --generate-results
#
# 3. Generate the graph PNGs:
#    python main.py --generate-vi-plot
#    python main.py --generate-mc-plot
#
# 4. Generate all files (results.txt + both PNGs):
#    python main.py --generate-all
