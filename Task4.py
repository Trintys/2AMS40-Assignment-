import numpy as np
import time

from Task1 import (
    XI_1, XI_2, GAMMA, DELTA, MAX_ITER,
    MODE_IDLE,
    create_state_space, create_state_index_map,
    get_valid_actions, get_next_states_and_probs, get_cost,
    precompute_transitions
)

# TASK 4: VALUE ITERATION (and optional policy iteration)


def value_iteration(states, state_index, transitions=None,
                    gamma=GAMMA, delta=DELTA, max_iter=MAX_ITER):

    n_states = len(states)
    V = np.zeros(n_states)
    pi = np.zeros(n_states, dtype=int)
    convergence_history = []

    # Precompute valid actions and one-step costs
    valid_actions_list = [get_valid_actions(state) for state in states]
    costs = {}
    for idx, state in enumerate(states):
        for action in valid_actions_list[idx]:
            costs[(idx, action)] = get_cost(state, action)

    for iteration in range(max_iter):
        V_new = np.zeros(n_states)
        max_diff = 0.0

        for idx, state in enumerate(states):
            valid_actions = valid_actions_list[idx]

            best_value = float('inf')
            best_action = valid_actions[0]

            for action in valid_actions:
                cost = costs[(idx, action)]

                # Expected future cost using precomputed transitions
                expected_future = 0.0
                if transitions is not None:
                    for next_idx, prob in transitions[(idx, action)]:
                        expected_future += prob * V[next_idx]
                else:
                    # Fallback (slower) if transitions not provided
                    trans_list = get_next_states_and_probs(state, action, states, state_index)
                    for next_state, prob in trans_list:
                        if next_state in state_index:
                            expected_future += prob * V[state_index[next_state]]

                value = cost + gamma * expected_future

                if value < best_value:
                    best_value = value
                    best_action = action

            V_new[idx] = best_value
            pi[idx] = best_action
            max_diff = max(max_diff, abs(V_new[idx] - V[idx]))

        V = V_new
        convergence_history.append(max_diff)

        if max_diff < delta:
            return V, pi, iteration + 1, convergence_history

    return V, pi, max_iter, convergence_history


def policy_iteration(states, state_index,
                     gamma=GAMMA, delta=DELTA, max_iter=MAX_ITER):

    n_states = len(states)
    V = np.zeros(n_states)

    # Initialize policy: first valid action for each state
    pi = np.zeros(n_states, dtype=int)
    for idx, state in enumerate(states):
        valid_actions = get_valid_actions(state)
        pi[idx] = valid_actions[0]

    for policy_iter in range(max_iter):
        # Policy Evaluation
        for eval_iter in range(max_iter):
            V_new = np.zeros(n_states)
            max_diff = 0.0

            for idx, state in enumerate(states):
                action = pi[idx]
                cost = get_cost(state, action)

                transitions = get_next_states_and_probs(state, action, states, state_index)
                expected_future = 0.0
                for next_state, prob in transitions:
                    if next_state in state_index:
                        expected_future += prob * V[state_index[next_state]]

                V_new[idx] = cost + gamma * expected_future
                max_diff = max(max_diff, abs(V_new[idx] - V[idx]))

            V = V_new
            if max_diff < delta:
                break

        # Policy Improvement
        policy_stable = True

        for idx, state in enumerate(states):
            valid_actions = get_valid_actions(state)
            old_action = pi[idx]

            best_value = float('inf')
            best_action = valid_actions[0]

            for action in valid_actions:
                cost = get_cost(state, action)

                transitions = get_next_states_and_probs(state, action, states, state_index)
                expected_future = 0.0
                for next_state, prob in transitions:
                    if next_state in state_index:
                        expected_future += prob * V[state_index[next_state]]

                value = cost + gamma * expected_future

                if value < best_value:
                    best_value = value
                    best_action = action

            pi[idx] = best_action
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            return V, pi, policy_iter + 1

    return V, pi, max_iter


def plot_convergence(convergence_history, delta=DELTA, save_path=None):

    import matplotlib.pyplot as plt

    iterations = range(1, len(convergence_history) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(iterations, convergence_history, 'b-', linewidth=2,
                label='Max value difference')
    ax.axhline(y=delta, color='r', linestyle='--', linewidth=2,
               label=f'Convergence threshold δ = {delta}')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Max |V_new(s) - V(s)|  (log scale)', fontsize=12)
    ax.set_title('Value Iteration Convergence', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    converge_iter = len(convergence_history)
    ax.annotate(
        f'Converged at iteration {converge_iter}',
        xy=(converge_iter, convergence_history[-1]),
        xytext=(max(1, int(converge_iter * 0.7)),
                convergence_history[-1] * 100),
        arrowprops=dict(arrowstyle='->', color='green'),
        fontsize=10,
        color='green'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")

    plt.close()
    return fig


def task4_value_iteration():

    print("\n\n\nTASK 4: VALUE ITERATION - OPTIMAL POLICY\n")

    start_time = time.time()

    # Create state space
    states = create_state_space()
    state_index = create_state_index_map(states)

    print(f"Total states: {len(states)}")
    print(f"Discount factor gamma = {GAMMA}")
    print(f"Convergence threshold delta = {DELTA}")

    # Precompute transitions for efficiency
    transitions = precompute_transitions(states, state_index)

    # Run value iteration
    V, pi, iterations, convergence = value_iteration(
        states, state_index, transitions=transitions
    )

    end_time = time.time()

    print(f"\nConverged in {iterations} iterations")
    print(f"Computation time: {end_time - start_time:.4f} seconds")

    # Convergence summary
    print(f"\nConvergence history (last 5 iterations):")
    for i, diff in enumerate(convergence[-5:], start=max(1, iterations - 4)):
        print(f"  Iteration {i}: max_diff = {diff:.2e}")

    # Initial state value: both healthy, engineer idle at depot
    initial_state = (0, 0, 0, MODE_IDLE, 0)
    initial_idx = state_index[initial_state]
    print(f"\nInitial state: {initial_state}")
    print(f"Optimal expected total discounted cost: {V[initial_idx]:.6f}")

    # Display optimal policy for depot states
    print("\nOPTIMAL POLICY (engineer at depot, idle):")
    print("State (s1, s2) | Action       | Value")

    action_names = {0: "Do nothing", 1: "Repair M1", 2: "Repair M2"}

    for s1 in range(XI_1 + 1):
        for s2 in range(XI_2 + 1):
            state = (s1, s2, 0, MODE_IDLE, 0)
            if state in state_index:
                idx = state_index[state]
                print(f"({s1}, {s2})          | {action_names[pi[idx]]:12} | {V[idx]:.4f}")

    # Idle-at-M1 states
    print("\nOPTIMAL POLICY (engineer idle at M1):")
    print("State (s1, s2) | Action       | Value")

    for s1 in range(XI_1 + 1):
        for s2 in range(XI_2 + 1):
            state = (s1, s2, 1, MODE_IDLE, 0)
            if state in state_index:
                idx = state_index[state]
                print(f"({s1}, {s2})          | {action_names[pi[idx]]:12} | {V[idx]:.4f}")

    # Idle-at-M2 states
    print("\nOPTIMAL POLICY (engineer idle at M2):")
    print("State (s1, s2) | Action       | Value")

    for s1 in range(XI_1 + 1):
        for s2 in range(XI_2 + 1):
            state = (s1, s2, 2, MODE_IDLE, 0)
            if state in state_index:
                idx = state_index[state]
                print(f"({s1}, {s2})          | {action_names[pi[idx]]:12} | {V[idx]:.4f}")

    return V, pi, states, state_index, iterations, convergence


# MODULE TEST

if __name__ == "__main__":
    V_optimal, pi_optimal, states, state_index, iterations, convergence = task4_value_iteration()