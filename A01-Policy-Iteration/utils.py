
directions = [
    (-1, 0),     # 0: up 
    (0, 1),      # 1: right
    (1, 0),      # 2: down
    (0, -1)      # 3: left
]
symbols = ['^', '>', 'v', '<']

def print_policy(size, env, policy):
    for r in range(size):
        for c in range(size):
            if (r, c) == env.goal_pos:
                print('G', end=' ')
            elif (r, c) in env.bads:
                print('X', end=' ')
            else:
                print(symbols[policy[r, c]], end=' ')
        print()
