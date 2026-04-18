import numpy as np
from scipy.stats import poisson


# -----------------------------------------------------------
# PARAMETERS FROM ASSIGNMENT
# -----------------------------------------------------------

XI_1 = 5      # failure threshold for machine 1
XI_2 = 7      # failure threshold for machine 2

LAMBDA = 0.5  # Poisson degradation rate
GAMMA = 0.9   # discount factor

# Costs
COST_PREVENTIVE = 1   # c_p
COST_CORRECTIVE = 5   # c_c
COST_UNAVAILABILITY = 1
COST_DO_NOTHING = 0

# Durations
TIME_TRAVEL = 1          # 1 time unit per leg (depot <-> machine)
TIME_PREVENTIVE = 1      # preventive repair duration
TIME_CORRECTIVE = 2      # corrective repair duration

# Engineer modes
MODE_IDLE = 0
MODE_TRAVEL = 1
MODE_REPAIR = 2

# Convergence parameters (used in Tasks 3 and 4)
DELTA = 1e-6
MAX_ITER = 10000

# Actions
# 0: do nothing / continue current job
# 1: start maintenance on machine 1
# 2: start maintenance on machine 2
ACTIONS = [0, 1, 2]


# -----------------------------------------------------------
# STATE SPACE
# -----------------------------------------------------------


def create_state_space():
    states = []
    for s1 in range(XI_1 + 1):
        for s2 in range(XI_2 + 1):

            # Idle states (engineer free) at depot, M1, M2
            for loc in [0, 1, 2]:
                states.append((s1, s2, loc, MODE_IDLE, 0))

            # Travel states: always from depot to a target machine
            # loc = 0 (depot), mode = TRAVEL, rem = target machine (1 or 2)
            for target in [1, 2]:
                states.append((s1, s2, 0, MODE_TRAVEL, target))

            # Repair states: at M1 or M2, remaining repair time 1 or 2
            for loc in [1, 2]:
                for rem in [1, 2]:
                    states.append((s1, s2, loc, MODE_REPAIR, rem))

    return states


def create_state_index_map(states):
    return {state: idx for idx, state in enumerate(states)}


# -----------------------------------------------------------
# VALID ACTIONS
# -----------------------------------------------------------


def get_valid_actions(state):
    s1, s2, loc, mode, rem = state

    if mode != MODE_IDLE:
        # travelling or repairing -> forced continuation
        return [0]

    # Mode idle
    m1_failed = (s1 == XI_1)
    m2_failed = (s2 == XI_2)

    if m1_failed or m2_failed:
        if m1_failed and not m2_failed:
            return [1]
        if m2_failed and not m1_failed:
            return [2]
        # both failed
        return [1, 2]

    # no failures
    return [0, 1, 2]


# -----------------------------------------------------------
# POISSON DEGRADATION KERNEL
# -----------------------------------------------------------


def poisson_probs(current_state, max_state, lam=LAMBDA):
    if current_state >= max_state:
        return {max_state: 1.0}

    probs = {}
    max_increment = max_state - current_state

    for k in range(max_increment):
        p = poisson.pmf(k, lam)
        if p > 1e-12:
            probs[current_state + k] = p

    # Tail probability: reaching or exceeding max_state
    p_tail = 1 - poisson.cdf(max_increment - 1, lam)
    if p_tail > 1e-12:
        probs[max_state] = p_tail

    return probs


# -----------------------------------------------------------
# TRANSITION DYNAMICS P(s' | s, a)
# -----------------------------------------------------------


def get_next_states_and_probs(state, action, states, state_index):
    s1, s2, loc, mode, rem = state
    transitions = {}

    # =========================================================
    # CASE 1: Engineer busy (mode != IDLE) -> action must be 0
    # =========================================================
    if mode != MODE_IDLE:
        if action != 0:
            # invalid combo, no transitions
            return []

        # -------------------------------
        # TRAVEL MODE: depot -> target
        # -------------------------------
        if mode == MODE_TRAVEL:
            target = rem  # 1 or 2

            # During travel, both machines degrade
            probs1 = poisson_probs(s1, XI_1)
            probs2 = poisson_probs(s2, XI_2)

            for next_s1, p1 in probs1.items():
                for next_s2, p2 in probs2.items():
                    prob = p1 * p2
                    if prob <= 1e-12:
                        continue

                    # After this step, we arrive at target machine
                    new_loc = target

                    # Decide repair time based on condition at arrival
                    if target == 1:
                        cond = next_s1
                        xi = XI_1
                    else:
                        cond = next_s2
                        xi = XI_2

                    repair_time = TIME_PREVENTIVE if cond < xi else TIME_CORRECTIVE

                    next_state = (next_s1, next_s2, new_loc, MODE_REPAIR, repair_time)
                    transitions[next_state] = transitions.get(next_state, 0.0) + prob

        # -------------------------------
        # REPAIR MODE: repairing machine 1 or 2
        # -------------------------------
        elif mode == MODE_REPAIR:
            target = loc  # 1 or 2

            if target == 1:
                # Machine 1 frozen, machine 2 degrades
                probs2 = poisson_probs(s2, XI_2)
                for next_s2, p2 in probs2.items():
                    prob = p2
                    if prob <= 1e-12:
                        continue

                    if rem == 1:
                        # Repair completes now: M1 becomes 0, engineer idle at M1
                        next_s1 = 0
                        next_state = (next_s1, next_s2, 1, MODE_IDLE, 0)
                    else:
                        # Repair continues: M1 state unchanged
                        next_s1 = s1
                        next_state = (next_s1, next_s2, 1, MODE_REPAIR, rem - 1)

                    transitions[next_state] = transitions.get(next_state, 0.0) + prob

            else:  # target == 2
                # Machine 2 frozen, machine 1 degrades
                probs1 = poisson_probs(s1, XI_1)
                for next_s1, p1 in probs1.items():
                    prob = p1
                    if prob <= 1e-12:
                        continue

                    if rem == 1:
                        # Repair completes now: M2 becomes 0, engineer idle at M2
                        next_s2 = 0
                        next_state = (next_s1, next_s2, 2, MODE_IDLE, 0)
                    else:
                        # Repair continues: M2 unchanged
                        next_s2 = s2
                        next_state = (next_s1, next_s2, 2, MODE_REPAIR, rem - 1)

                    transitions[next_state] = transitions.get(next_state, 0.0) + prob

    # =========================================================
    # CASE 2: Engineer idle (mode == IDLE)
    # =========================================================
    else:
        # safety: rem should be 0 in idle states

        # -------------------------------
        # Action 0: do nothing
        # -------------------------------
        if action == 0:
            probs1 = poisson_probs(s1, XI_1)
            probs2 = poisson_probs(s2, XI_2)

            for next_s1, p1 in probs1.items():
                for next_s2, p2 in probs2.items():
                    prob = p1 * p2
                    if prob <= 1e-12:
                        continue
                    # engineer stays where she is, still idle
                    next_state = (next_s1, next_s2, loc, MODE_IDLE, 0)
                    transitions[next_state] = transitions.get(next_state, 0.0) + prob

        # -------------------------------
        # Action 1: start maintenance on machine 1
        # -------------------------------
        elif action == 1:
            target = 1

            if loc == 1:
                # Already at M1: no travel, start repair immediately
                # M1 frozen, M2 degrades
                repair_time = TIME_PREVENTIVE if s1 < XI_1 else TIME_CORRECTIVE
                probs2 = poisson_probs(s2, XI_2)

                for next_s2, p2 in probs2.items():
                    prob = p2
                    if prob <= 1e-12:
                        continue

                    if repair_time == 1:
                        # single-step repair finishes now
                        next_s1 = 0
                        next_state = (next_s1, next_s2, 1, MODE_IDLE, 0)
                    else:
                        # two-step repair: one step consumed, one remains
                        next_s1 = s1
                        next_state = (next_s1, next_s2, 1, MODE_REPAIR, repair_time - 1)

                    transitions[next_state] = transitions.get(next_state, 0.0) + prob

            elif loc == 0:
                # From depot -> M1 (one travel time unit)
                probs1 = poisson_probs(s1, XI_1)
                probs2 = poisson_probs(s2, XI_2)

                for next_s1, p1 in probs1.items():
                    for next_s2, p2 in probs2.items():
                        prob = p1 * p2
                        if prob <= 1e-12:
                            continue

                        # After this step we arrive at M1; repair starts NEXT step,
                        # so we enter a repair state with full repair_time.
                        cond = next_s1
                        xi = XI_1
                        repair_time = TIME_PREVENTIVE if cond < xi else TIME_CORRECTIVE

                        next_state = (next_s1, next_s2, 1, MODE_REPAIR, repair_time)
                        transitions[next_state] = transitions.get(next_state, 0.0) + prob

            else:
                # loc == 2 : at M2, want to repair M1
                # First leg: M2 -> depot (1 time unit), both machines degrade
                probs1 = poisson_probs(s1, XI_1)
                probs2 = poisson_probs(s2, XI_2)

                for next_s1, p1 in probs1.items():
                    for next_s2, p2 in probs2.items():
                        prob = p1 * p2
                        if prob <= 1e-12:
                            continue

                        # After this step: at depot, we still need depot -> M1 (one travel unit).
                        # Represent second leg as MODE_TRAVEL from depot to target=1.
                        next_state = (next_s1, next_s2, 0, MODE_TRAVEL, target)
                        transitions[next_state] = transitions.get(next_state, 0.0) + prob

        # -------------------------------
        # Action 2: start maintenance on machine 2
        # -------------------------------
        elif action == 2:
            target = 2

            if loc == 2:
                # Already at M2: no travel, start repair immediately
                # M2 frozen, M1 degrades
                repair_time = TIME_PREVENTIVE if s2 < XI_2 else TIME_CORRECTIVE
                probs1 = poisson_probs(s1, XI_1)

                for next_s1, p1 in probs1.items():
                    prob = p1
                    if prob <= 1e-12:
                        continue

                    if repair_time == 1:
                        # single-step repair finishes now
                        next_s2 = 0
                        next_state = (next_s1, next_s2, 2, MODE_IDLE, 0)
                    else:
                        next_s2 = s2
                        next_state = (next_s1, next_s2, 2, MODE_REPAIR, repair_time - 1)

                    transitions[next_state] = transitions.get(next_state, 0.0) + prob

            elif loc == 0:
                # From depot -> M2 (one travel unit)
                probs1 = poisson_probs(s1, XI_1)
                probs2 = poisson_probs(s2, XI_2)

                for next_s1, p1 in probs1.items():
                    for next_s2, p2 in probs2.items():
                        prob = p1 * p2
                        if prob <= 1e-12:
                            continue

                        cond = next_s2
                        xi = XI_2
                        repair_time = TIME_PREVENTIVE if cond < xi else TIME_CORRECTIVE

                        next_state = (next_s1, next_s2, 2, MODE_REPAIR, repair_time)
                        transitions[next_state] = transitions.get(next_state, 0.0) + prob

            else:
                # loc == 1 : at M1, want to repair M2
                # First leg: M1 -> depot (1 time unit), both degrade
                probs1 = poisson_probs(s1, XI_1)
                probs2 = poisson_probs(s2, XI_2)

                for next_s1, p1 in probs1.items():
                    for next_s2, p2 in probs2.items():
                        prob = p1 * p2
                        if prob <= 1e-12:
                            continue

                        # After this step we are at depot, still need depot->M2
                        next_state = (next_s1, next_s2, 0, MODE_TRAVEL, target)
                        transitions[next_state] = transitions.get(next_state, 0.0) + prob

    # Convert dict to list
    return list(transitions.items())


# -----------------------------------------------------------
# ONE-STEP COST c(s,a) and REWARD R(s,a) = -c(s,a)
# -----------------------------------------------------------


def get_cost(state, action):
    s1, s2, loc, mode, rem = state
    cost = 0.0

    # Unavailability: machine is unavailable if
    # - it is failed (state xi_i)
    # - or it is being repaired (mode == REPAIR and loc = that machine)
    # Travel does NOT count as repair; only REPAIR mode.
    if s1 == XI_1 or (mode == MODE_REPAIR and loc == 1):
        cost += COST_UNAVAILABILITY
    if s2 == XI_2 or (mode == MODE_REPAIR and loc == 2):
        cost += COST_UNAVAILABILITY

    # Maintenance cost is charged only when starting maintenance
    # from an idle state (mode = IDLE and action ∈ {1,2}).
    if mode == MODE_IDLE and action in [1, 2]:
        if action == 1:
            # M1: preventive or corrective
            cost += COST_PREVENTIVE if s1 < XI_1 else COST_CORRECTIVE
        elif action == 2:
            # M2
            cost += COST_PREVENTIVE if s2 < XI_2 else COST_CORRECTIVE

    # Action 0 from idle has no extra cost (only unavailability above)
    return cost


def get_reward(state, action):
    return -get_cost(state, action)


# -----------------------------------------------------------
# OPTIONAL: PRECOMPUTE TRANSITIONS (for later tasks)
# -----------------------------------------------------------


def precompute_transitions(states, state_index):
    transitions = {}
    for idx, state in enumerate(states):
        valid_actions = get_valid_actions(state)
        for action in valid_actions:
            next_list = get_next_states_and_probs(state, action, states, state_index)
            indexed = [(state_index[ns], p) for ns, p in next_list if ns in state_index]
            transitions[(idx, action)] = indexed
    return transitions


# -----------------------------------------------------------
# SIMPLE SELF-TEST
# -----------------------------------------------------------

if __name__ == "__main__":
    print(f"TASK 1: MDP DEFINITION\nMDP Parameters:")
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
    
    states = create_state_space()
    state_index = create_state_index_map(states)
    print(f"\nTotal states: {len(states)}")
    print(f"Actions: {ACTIONS}")
    
    # Test valid actions
    print("\nSample valid actions:")
    test_states = [
        (0, 0, 0, 0),  # Both healthy, engineer at depot
        (0, 0, 1, 0),  # Both healthy, engineer idle at M1
        (0, 0, 2, 0),  # Both healthy, engineer idle at M2
        (5, 0, 0, 0),  # M1 failed, engineer at depot
        (5, 0, 1, 0),  # M1 failed, engineer idle at M1
        (0, 7, 2, 0),  # M2 failed, engineer idle at M2
        (5, 7, 0, 0),  # Both failed, engineer at depot
        (3, 4, 1, 2),  # Engineer busy on M1
    ]
    for s in test_states:
        print(f"  State {s}: valid actions = {get_valid_actions(s)}")
    
    # Test transitions from depot (with travel)
    print("\nSample transitions from (0, 0, 0, 0) with action 1 (dispatch to M1 from depot):")
    trans = get_next_states_and_probs((0, 0, 0, 0), 1, states, state_index)
    for next_state, prob in sorted(trans, key=lambda x: -x[1])[:5]:
        print(f"  -> {next_state}: {prob:.4f}")
    
    # Test transitions from same machine (no travel)
    print("\nSample transitions from (2, 3, 1, 0) with action 1 (repair M1 while at M1 - no travel):")
    trans = get_next_states_and_probs((2, 3, 1, 0), 1, states, state_index)
    for next_state, prob in sorted(trans, key=lambda x: -x[1])[:5]:
        print(f"  -> {next_state}: {prob:.4f}")
    
    # Test that engineer stays at machine after repair
    print("\nSample transitions from (2, 3, 1, 1) - repair completing (engineer stays at M1):")
    trans = get_next_states_and_probs((2, 3, 1, 1), 0, states, state_index)
    for next_state, prob in sorted(trans, key=lambda x: -x[1])[:5]:
        print(f"  -> {next_state}: {prob:.4f}")
