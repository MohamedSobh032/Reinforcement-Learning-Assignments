import numpy as np
from utils import directions, print_policy

def policy_iteration(env, gamma, theta, max_iters):
    """
    ==================================================
     Policy Iteration (Dynamic Programming)
    ==================================================
     Performs iterative policy evaluation and improvement
     on a known grid-world MDP until the policy stabilizes.
    ==================================================
    """

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    size = env.grid_size
    policy = np.random.choice(4, size=(size, size))  # Random initial policy
    value = np.zeros((size, size))
    iterations = 0

    # --------------------------------------------------
    # Helper Functions
    # --------------------------------------------------
    def is_terminal(s):
        """Check if a state is terminal (goal or bad cell)."""
        return s == env.goal_pos or tuple(s) in env.bads

    def get_next_pos(s, d):
        """Compute the next state given current state and movement delta."""
        nr, nc = s[0] + d[0], s[1] + d[1]
        if 0 <= nr < size and 0 <= nc < size:
            return (nr, nc)
        return s  # Stay in place if hitting wall

    def get_transitions(a):
        """Return possible movement directions and their probabilities."""
        intended = directions[a]
        perp1 = directions[(a - 1) % 4]
        perp2 = directions[(a + 1) % 4]
        return [
            (intended, env.prob_intended),
            (perp1, env.prob_perp),
            (perp2, env.prob_perp)
        ]

    # --------------------------------------------------
    # Policy Iteration Loop
    # --------------------------------------------------
    policy_stable = False
    iters = 0
    while not policy_stable and iters < max_iters:
        iterations += 1
        print("=" * 60)
        print(f"[Iteration {iterations}] Policy Evaluation Phase")
        print("=" * 60)

        # ----------------------------------------------
        # Policy Evaluation
        # ----------------------------------------------
        while True:
            delta = 0
            for r in range(size):
                for c in range(size):
                    s = (r, c)
                    if is_terminal(s):
                        value[r, c] = 0
                        continue

                    v_old = value[r, c]
                    a = policy[r, c]
                    v_new = 0

                    # Expected value under current policy
                    for direction_delta, prob in get_transitions(a):
                        ns = get_next_pos(s, direction_delta)
                        rwd = env.get_reward(ns)
                        v_new += prob * (
                            rwd + gamma * value[ns[0], ns[1]] if not is_terminal(ns) else rwd
                        )

                    value[r, c] = v_new
                    delta = max(delta, abs(v_old - v_new))

            if delta < theta:
                break

        print(f"  → Policy Evaluation converged (Δ < {theta})")

        # ----------------------------------------------
        # Policy Improvement
        # ----------------------------------------------
        print(f"[Iteration {iterations}] Policy Improvement Phase")
        policy_stable = True

        for r in range(size):
            for c in range(size):
                s = (r, c)
                if is_terminal(s):
                    continue

                old_a = policy[r, c]
                q_values = np.zeros(4)

                for a in range(4):
                    for direction_delta, prob in get_transitions(a):
                        ns = get_next_pos(s, direction_delta)
                        rwd = env.get_reward(ns)
                        q_values[a] += prob * (
                            rwd + gamma * value[ns[0], ns[1]] if not is_terminal(ns) else rwd
                        )

                new_a = np.argmax(q_values)
                policy[r, c] = new_a

                if new_a != old_a:
                    policy_stable = False

        iters += 1
        print("  → Policy stable:", policy_stable)
        print()

    # --------------------------------------------------
    # Final Policy and Summary
    # --------------------------------------------------
    print("=" * 60)
    print("✅ POLICY ITERATION COMPLETE")
    print("=" * 60)
    print(f"Converged after {iterations} iterations.\n")

    print("Final Policy (Grid Arrows):")
    print_policy(size=size, env=env, policy=policy)

    return policy, iterations
