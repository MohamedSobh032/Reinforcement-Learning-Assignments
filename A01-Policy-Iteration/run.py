import argparse
import datetime
from env import GridMazeEnv
from policy_iteration_dp import policy_iteration
from gymnasium.wrappers import RecordVideo
import os

# ==================================================
# Argument Parser
# ==================================================
parser = argparse.ArgumentParser(description="Run Policy Iteration on Grid Maze Environment")

# Environment Configuration
parser.add_argument('--grid_size', type=int, default=5, help='Grid size (e.g., 5 for 5x5)')
parser.add_argument('--bads', type=int, default=2, help='Number of bad cells in the maze')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

# Experiment Parameters
parser.add_argument('--prob_intended', type=float, default=0.7, help='Probability of intended move')
parser.add_argument('--prob_perp', type=float, default=0.15, help='Probability of perpendicular move')
parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
parser.add_argument('--theta', type=float, default=1e-6, help='Convergence threshold')
parser.add_argument('--reward_goal', type=float, default=10, help='Reward for reaching goal')
parser.add_argument('--reward_bad', type=float, default=-10, help='Penalty for landing on bad cell')
parser.add_argument('--reward_step', type=float, default=-1, help='Penalty per step to encourage shorter paths')
parser.add_argument('--max_iters', type=int, default=1000, help='Maximum policy iteration cycles')

# Rendering & Output
parser.add_argument('--timestep', type=int, default=200, help='Time between moves in milliseconds')
args = parser.parse_args()

# ==================================================
# Setup Output Directory and Logging
# ==================================================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = f"videos/run_{timestamp}"
os.makedirs(video_path, exist_ok=True)

print("\n" + "=" * 60)
print("🎬 Reinforcement Learning — Policy Iteration (DP)")
print("=" * 60)
print(f"[INFO] Experiment started at: {timestamp}")
print(f"[INFO] Grid size: {args.grid_size}x{args.grid_size}, Bad cells: {args.bads}")
print(f"[INFO] Probabilities -> Intended: {args.prob_intended}, Perpendicular: {args.prob_perp}")
print(f"[INFO] Rewards -> Goal: {args.reward_goal}, Bad: {args.reward_bad}, Step: {args.reward_step}")
print(f"[INFO] Gamma: {args.gamma}, Theta: {args.theta}")
print(f"[INFO] Video output: {video_path}")
print("=" * 60 + "\n")

# ==================================================
# Environment Initialization
# ==================================================
env = GridMazeEnv(
    grid_size=args.grid_size,
    num_bads=args.bads,
    seed=args.seed,
    prob_intended=args.prob_intended,
    prob_perp=args.prob_perp,
    reward_goal=args.reward_goal,
    reward_bad=args.reward_bad,
    reward_step=args.reward_step
)

env.render_mode = 'rgb_array'
timestep_ms = max(args.timestep, 1)
fps = max(1.0, 1000.0 / timestep_ms)
env.metadata["render_fps"] = fps

# ==================================================
# Run Policy Iteration
# ==================================================
print("🚀 Running Policy Iteration...")
policy, converged_iterations = policy_iteration(env, args.gamma, args.theta)
print(f"✅ Policy Iteration converged in {converged_iterations} iterations.\n")

# ==================================================
# Record Simulation Video
# ==================================================
print("🎞️ Recording agent trajectory...")
video_env = RecordVideo(
    env,
    video_folder=video_path,
    episode_trigger=lambda x: True,
    name_prefix='grid_maze',
    disable_logger=True
)

obs, _ = video_env.reset()
terminated = truncated = False
steps = 0

while not (terminated or truncated):
    agent_pos = video_env.env.agent_pos
    action = policy[agent_pos[0], agent_pos[1]]
    obs, reward, terminated, truncated, _ = video_env.step(action)
    steps += 1
    print(f"[Step {steps:03d}] Action={action:<2} | Pos={agent_pos} | Reward={reward:+4.1f} | Done={terminated}")

video_env.close()

# ==================================================
# Final Summary
# ==================================================
print("\n" + "=" * 60)
print("🏁 EXPERIMENT SUMMARY")
print("=" * 60)
print(f"Experiment Timestamp : {timestamp}")
print(f"Grid Size            : {args.grid_size}x{args.grid_size}")
print(f"Bad Cells            : {args.bads}")
print(f"Converged In         : {converged_iterations} Policy Iterations")
print(f"Reached Goal In      : {steps} Simulation Steps")
print(f"Video Output Path    : {os.path.abspath(video_path)}")
print("=" * 60 + "\n")
print("🎉 Simulation Complete.\n")
