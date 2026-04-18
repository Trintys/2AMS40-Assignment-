import numpy as np
import time

from Task1 import (
    XI_1, XI_2, LAMBDA, GAMMA, DELTA, MAX_ITER,
    COST_UNAVAILABILITY,
    TIME_TRAVEL, TIME_PREVENTIVE, TIME_CORRECTIVE,
    MODE_IDLE,
    create_state_space, create_state_index_map,
    get_next_states_and_probs, get_cost, get_valid_actions,
    precompute_transitions
)

# TASK 3: TOTAL EXPECTED DISCOUNTED COST - "CORRECTIVE-ONLY" POLICY


def failure_only_policy(state):

    s1, s2, loc, mode, rem = state

    # Engineer busy (travelling or repairing) -> must wait
    if mode != MODE_IDLE:
        return 0

    # Only repair at failure
    if s1 == XI_1:
        return 1
    elif s2 == XI_2:
        return 2
    else:
        return 0  # Do nothing (no failures)


def evaluate_policy(policy_func, states, state_index, transitions=None,
                    gamma=GAMMA, delta=DELTA, max_iter=MAX_ITER):

    n_states = len(states)
    V = np.zeros(n_states)

    # Precompute policy actions for all states
    policy_actions = [policy_func(state) for state in states]

    # Validate actions and compute costs once
    costs = np.zeros(n_states)
    for idx, state in enumerate(states):
        action = policy_actions[idx]
        valid_actions = get_valid_actions(state)
        if action not in valid_actions:
            policy_actions[idx] = valid_actions[0]
            action = valid_actions[0]
        costs[idx] = get_cost(state, action)

    for iteration in range(max_iter):
        V_new = np.zeros(n_states)
        max_diff = 0.0

        for idx, state in enumerate(states):
            action = policy_actions[idx]

            # Expected future cost
            expected_future = 0.0
            if transitions is not None:
                for next_idx, prob in transitions[(idx, action)]:
                    expected_future += prob * V[next_idx]
            else:
                trans_list = get_next_states_and_probs(state, action, None, None)
                for next_state, prob in trans_list:
                    if next_state in state_index:
                        expected_future += prob * V[state_index[next_state]]

            V_new[idx] = costs[idx] + gamma * expected_future
            max_diff = max(max_diff, abs(V_new[idx] - V[idx]))

        V = V_new
        if max_diff < delta:
            return V, iteration + 1

    return V, max_iter


def simulate_policy(policy_func, initial_state, n_steps, gamma=GAMMA):
    state = initial_state
    total_discounted_cost = 0.0
    discount = 1.0

    for t in range(n_steps):
        # Get action from policy
        action = policy_func(state)
        valid_actions = get_valid_actions(state)
        if action not in valid_actions:
            action = valid_actions[0]

        # Immediate cost
        cost = get_cost(state, action)
        total_discounted_cost += discount * cost
        discount *= gamma

        if discount < 1e-10:
            break

        # Sample next state from P(s'|s,a)
        trans_list = get_next_states_and_probs(state, action, None, None)
        if not trans_list:
            break

        next_states, probs = zip(*trans_list)
        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum()
        idx = np.random.choice(len(next_states), p=probs)
        state = next_states[idx]

    return total_discounted_cost


def monte_carlo_estimate(policy_func, initial_state,
                         n_simulations=10000, n_steps=500,
                         gamma=GAMMA, seed=42):
    
    np.random.seed(seed)
    costs = []

    for _ in range(n_simulations):
        cost = simulate_policy(policy_func, initial_state, n_steps, gamma)
        costs.append(cost)

    costs = np.array(costs)
    mean = np.mean(costs)
    std = np.std(costs, ddof=1)
    se = std / np.sqrt(n_simulations)
    ci_lower = mean - 1.96 * se
    ci_upper = mean + 1.96 * se

    return mean, se, ci_lower, ci_upper


def task3_evaluate_failure_only():

    print("\n\n\nTASK 3: EVALUATE 'CORRECTIVE-ONLY' (FAILURE-ONLY) POLICY\n")

    start_time = time.time()

    # Create state space
    states = create_state_space()
    state_index = create_state_index_map(states)

    print(f"Total states: {len(states)}")

    # Precompute transitions for efficiency
    transitions = precompute_transitions(states, state_index)

    # Evaluate the policy
    V, iterations = evaluate_policy(
        failure_only_policy, states, state_index, transitions=transitions
    )

    # Initial state: both healthy, engineer idle at depot
    initial_state = (0, 0, 0, MODE_IDLE, 0)
    initial_idx = state_index[initial_state]

    end_time = time.time()

    print(f"Policy evaluation converged in {iterations} iterations")
    print(f"Convergence threshold: delta = {DELTA}")
    print(f"\nInitial state: {initial_state}")
    print(f"Expected total discounted cost (corrective-only): {V[initial_idx]:.6f}")
    print(f"\nResult type: EXACT (analytical solution via Bellman equations)")
    print(f"Computation time: {end_time - start_time:.4f} seconds")

    # Value function for selected states
    print("\nValue function for selected states:")
    sample_states = [
        (0, 0, 0, MODE_IDLE, 0),
        (0, 0, 1, MODE_IDLE, 0),
        (0, 0, 2, MODE_IDLE, 0),
        (3, 3, 0, MODE_IDLE, 0),
        (5, 0, 0, MODE_IDLE, 0),
        (0, 7, 0, MODE_IDLE, 0),
        (5, 7, 0, MODE_IDLE, 0),
    ]
    for s in sample_states:
        if s in state_index:
            print(f"  V{s} = {V[state_index[s]]:.6f}")

    return V, states, state_index, iterations


def plot_mc_convergence(n_simulations_range, n_steps=500, seed=42, save_path=None):
    import matplotlib.pyplot as plt

    np.random.seed(seed)

    max_sims = max(n_simulations_range)
    all_costs = []

    initial_state = (0, 0, 0, MODE_IDLE, 0)

    for _ in range(max_sims):
        cost = simulate_policy(failure_only_policy, initial_state, n_steps, GAMMA)
        all_costs.append(cost)

    all_costs = np.array(all_costs)

    running_means = np.cumsum(all_costs) / np.arange(1, max_sims + 1)
    running_stds = [np.std(all_costs[:i+1], ddof=1) if i > 0 else 0 for i in range(max_sims)]
    running_ses = [running_stds[i] / np.sqrt(i+1) for i in range(max_sims)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(1, max_sims + 1)
    ax1.plot(x, running_means, linewidth=1.5, label='Running mean')
    ax1.axhline(y=running_means[-1], color='g', linestyle='--', alpha=0.7,
                label=f'Final estimate: {running_means[-1]:.4f}')
    ax1.fill_between(
        x,
        [running_means[i] - 1.96 * running_ses[i] for i in range(max_sims)],
        [running_means[i] + 1.96 * running_ses[i] for i in range(max_sims)],
        alpha=0.3, label='95% CI'
    )
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Estimated Expected Cost')
    ax1.set_title('Monte Carlo Estimate Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    ax2.loglog(x[9:], running_ses[9:], linewidth=1.5, label='Standard error')
    ax2.axhline(y=0.03, color='r', linestyle='--', linewidth=2, label='Target SE = 0.03')
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('Standard Error (log scale)')
    ax2.set_title('Standard Error Reduction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    idx_10k = min(9999, max_sims - 1)
    ax2.scatter([10000], [running_ses[idx_10k]])
    ax2.annotate(
        f'n=10000, SE={running_ses[idx_10k]:.4f}',
        xy=(10000, running_ses[idx_10k]),
        xytext=(3000, running_ses[idx_10k] * 3),
        arrowprops=dict(arrowstyle='->')
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Monte Carlo convergence plot saved to: {save_path}")

    plt.close()
    return fig


def task3_monte_carlo_verification(generate_plot=False, plot_path=None):

    print("\n\n\nTASK 3 VERIFICATION: MONTE CARLO SIMULATION\n")

    start_time = time.time()

    n_simulations = 10000
    n_steps = 500

    print(f"Number of simulations: {n_simulations}")
    print(f"Steps per simulation: {n_steps}")
    print(f"Discount factor gamma = {GAMMA}")

    initial_state = (0, 0, 0, MODE_IDLE, 0)

    mean, se, ci_lower, ci_upper = monte_carlo_estimate(
        failure_only_policy,
        initial_state=initial_state,
        n_simulations=n_simulations,
        n_steps=n_steps
    )

    end_time = time.time()

    print(f"\nMonte Carlo estimate: {mean:.6f}")
    print(f"Standard error: {se:.6f}")
    print(f"95% Confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"Computation time: {end_time - start_time:.4f} seconds")

    print("\nJustification for Monte Carlo parameters:")
    print(f"  - n_simulations = {n_simulations}: gives SE ~ {se:.4f}")
    print(f"  - n_steps = {n_steps}: sufficient since gamma^{n_steps} ≈ {GAMMA**n_steps:.2e}")

    if generate_plot:
        print("\nGenerating Monte Carlo convergence plot...")
        plot_mc_convergence([100, 500, 1000, 5000, 10000],
                            n_steps=n_steps, save_path=plot_path)

    return mean, se, ci_lower, ci_upper


# MODULE TEST

if __name__ == "__main__":
    V_failure, states, state_index, iterations = task3_evaluate_failure_only()
    task3_monte_carlo_verification()