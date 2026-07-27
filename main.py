import numpy as np


# States
states = ["S0", "S1", "S2", "S3"]
num_states = len(states)

# Actions
LEFT = 0
RIGHT = 1
actions = [LEFT, RIGHT]
action_names = {LEFT: "LEFT", RIGHT: "RIGHT"}

gamma = 0.9

# Transition model:
# transition[state][action] = (next_state, reward)
transition = {
    0: {
        RIGHT: (1, -1)
    },

    1: {
        LEFT: (0, -1),
        RIGHT: (2, -1)
    },

    2: {
        LEFT: (1, -1),
        RIGHT: (3, 10)
    },

    3: {}
}

terminal_state = 3


policy = {
    0: RIGHT,
    1: LEFT,
    2: RIGHT
}

# Initialize Value Function
V = np.zeros(num_states)

# Policy Evaluation
def policy_evaluation(policy, V, theta=1e-6):

    while True:

        delta = 0

        for s in range(num_states):

            if s == terminal_state:
                continue

            action = policy[s]

            next_state, reward = transition[s][action]

            new_value = reward + gamma * V[next_state]

            delta = max(delta, abs(new_value - V[s]))

            V[s] = new_value

        if delta < theta:
            break

    return V


# Policy Improvement
def policy_improvement(policy, V):

    policy_stable = True

    for s in range(num_states):

        if s == terminal_state:
            continue

        old_action = policy[s]

        best_action = old_action
        best_value = -np.inf

        # Evaluate every possible action
        for action in transition[s]:

            next_state, reward = transition[s][action]

            q = reward + gamma * V[next_state]

            if q > best_value:
                best_value = q
                best_action = action

        policy[s] = best_action

        if best_action != old_action:
            policy_stable = False

    return policy, policy_stable


# Policy Iteration
iteration = 1

while True:

    print("=" * 50)
    print(f"Iteration {iteration}")

    V = policy_evaluation(policy, V)

    print("\nValue Function")
    for i in range(num_states):
        print(f"{states[i]} : {V[i]:.3f}")

    policy, stable = policy_improvement(policy, V)

    print("\nPolicy")
    for s in range(terminal_state):
        print(f"{states[s]} -> {action_names[policy[s]]}")

    if stable:
        print("\nPolicy converged.")
        break

    iteration += 1


# Final Result
print("\n" + "=" * 50)
print("Optimal Value Function")

for i in range(num_states):
    print(f"{states[i]} : {V[i]:.3f}")

print("\nOptimal Policy")

for s in range(terminal_state):
    print(f"{states[s]} -> {action_names[policy[s]]}")