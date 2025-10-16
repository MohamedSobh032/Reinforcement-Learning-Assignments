import gymnasium as gym
from gymnasium.utils.seeding import np_random
import pygame
import numpy as np
import itertools
from utils import directions

# ==================================================
# Color Configuration
# ==================================================
COLOR_BACKGROUND = (255, 255, 255)   # White
COLOR_GRID_LINES = (0, 0, 0)         # Black
COLOR_GOAL       = (0, 255, 0)       # Green
COLOR_BAD        = (255, 0, 0)       # Red
COLOR_AGENT      = (0, 0, 255)       # Blue
COLOR_BORDER_WIDTH = 3               # Grid line thickness


# ==================================================
# Grid Maze Environment Definition
# ==================================================
class GridMazeEnv(gym.Env):
    """A simple stochastic grid-based maze environment for dynamic programming and RL experiments."""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def __init__(self, grid_size, num_bads, seed, prob_intended, prob_perp,
                 reward_goal, reward_bad, reward_step):
        super().__init__()

        # Basic dimensions and structure
        self.grid_size = grid_size
        self.num_bads = num_bads
        self.observation_space = gym.spaces.Tuple([gym.spaces.Discrete(self.grid_size)] * 8)
        self.action_space = gym.spaces.Discrete(4)

        # Experimental configuration
        self.prob_intended = prob_intended
        self.prob_perp = prob_perp
        self.reward_goal = reward_goal
        self.reward_bad = reward_bad
        self.reward_step = reward_step

        # Proper seeding
        self.np_random, self.seed = np_random(seed)

        # Generate fixed positions for goal and bad cells
        all_pos = list(itertools.product(range(self.grid_size), range(self.grid_size)))
        indices = self.np_random.choice(len(all_pos), self.num_bads + 1, replace=False)
        self.goal_pos = all_pos[indices[0]]
        self.bads = {all_pos[i] for i in indices[1:]}
        self.agent_pos = None

        # Pygame setup
        self.cell_size = 50
        self.window_size = (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
        self.window = None
        self.clock = None

        print("=" * 60)
        print("Environment Initialized:")
        print(f"  Grid Size     : {self.grid_size}x{self.grid_size}")
        print(f"  Goal Position : {self.goal_pos}")
        print(f"  Bad Cells     : {sorted(self.bads)}")
        print(f"  Rewards       : Goal={self.reward_goal}, Bad={self.reward_bad}, Step={self.reward_step}")
        print(f"  Probabilities : Intended={self.prob_intended}, Perp={self.prob_perp}")
        print("=" * 60)

    # --------------------------------------------------
    # Reward Function
    # --------------------------------------------------
    def get_reward(self, next_s):
        """Return the reward associated with transitioning into the next state."""
        if next_s == self.goal_pos:
            return self.reward_goal
        if tuple(next_s) in self.bads:
            return self.reward_bad
        return self.reward_step

    # --------------------------------------------------
    # Observation Construction
    # --------------------------------------------------
    def get_observation(self):
        """Return a flattened state representation (agent, goal, and bad cell positions)."""
        bads_flat = list(itertools.chain.from_iterable(self.bads))
        return tuple(list(self.agent_pos) + list(self.goal_pos) + bads_flat)

    # --------------------------------------------------
    # Environment Step
    # --------------------------------------------------
    def step(self, action):
        """Perform one stochastic movement step based on action probabilities."""
        intended = directions[action]
        perp1 = directions[(action - 1) % 4]
        perp2 = directions[(action + 1) % 4]
        choice = self.np_random.choice([0, 1, 2], p=[self.prob_intended, self.prob_perp, self.prob_perp])

        # Select movement direction
        delta = intended if choice == 0 else (perp1 if choice == 1 else perp2)
        new_row = self.agent_pos[0] + delta[0]
        new_col = self.agent_pos[1] + delta[1]

        # Update position if within bounds
        if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
            self.agent_pos = (new_row, new_col)

        obs = self.get_observation()
        print(f"  → Step: Action={action}, NewPos={self.agent_pos}, Reward={self.get_reward(self.agent_pos)}")
        return obs, self.get_reward(self.agent_pos), self.agent_pos == self.goal_pos, False, {}

    # --------------------------------------------------
    # Environment Reset
    # --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """Reset agent to a random position sufficiently far from the goal."""
        super().reset(seed=self.seed)
        min_distance = max(2, self.grid_size // 2)

        while True:
            pos_idx = self.np_random.integers(0, self.grid_size ** 2)
            pos = divmod(pos_idx, self.grid_size)
            distance = abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])
            if pos != self.goal_pos and pos not in self.bads and distance >= min_distance:
                self.agent_pos = pos
                break

        print(f"  → Environment reset. Agent start position: {self.agent_pos} (distance={distance})")
        return self.get_observation(), {}

    # --------------------------------------------------
    # Rendering Function
    # --------------------------------------------------
    def render(self):
        """Render the current grid using pygame."""
        canvas = pygame.Surface(self.window_size)
        canvas.fill(COLOR_BACKGROUND)

        # Draw grid lines
        for x in range(self.grid_size + 1):
            pygame.draw.line(canvas, COLOR_GRID_LINES, (0, x * self.cell_size),
                             (self.window_size[0], x * self.cell_size), width=COLOR_BORDER_WIDTH)
            pygame.draw.line(canvas, COLOR_GRID_LINES, (x * self.cell_size, 0),
                             (x * self.cell_size, self.window_size[1]), width=COLOR_BORDER_WIDTH)

        # Draw goal cell
        pygame.draw.rect(canvas, COLOR_GOAL,
                         pygame.Rect(self.goal_pos[1] * self.cell_size, self.goal_pos[0] * self.cell_size,
                                     self.cell_size, self.cell_size))

        # Draw bad cells
        for bad in self.bads:
            pygame.draw.rect(canvas, COLOR_BAD,
                             pygame.Rect(bad[1] * self.cell_size, bad[0] * self.cell_size,
                                         self.cell_size, self.cell_size))

        # Draw agent
        pygame.draw.rect(canvas, COLOR_AGENT,
                         pygame.Rect(self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size,
                                     self.cell_size, self.cell_size))

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    # --------------------------------------------------
    # Environment Cleanup
    # --------------------------------------------------
    def close(self):
        """Properly close the rendering window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
