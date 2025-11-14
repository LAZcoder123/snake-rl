import os
import random
import time
import pickle
from collections import defaultdict

WIDTH, HEIGHT = 10, 10
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

class SnakeGame:
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.score = 0
        self.food = self._random_empty_cell()
        self.bad_block = self._random_empty_cell(exclude={self.food})
        return self.get_state()

    def _random_empty_cell(self, exclude=None):
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)
        exclude = exclude.union(set(self.snake))
        while True:
            cell = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if cell not in exclude:
                return cell

    def get_state(self):
        head_x, head_y = self.snake[0]
        fx, fy = self.food
        bx, by = self.bad_block

        def rel(val):
            if val < 0: return -1
            elif val > 0: return 1
            else: return 0

        return (rel(fx - head_x), rel(fy - head_y), rel(bx - head_x), rel(by - head_y))

    def step(self, action):
        head_x, head_y = self.snake[0]

        if action == "UP": head_y -= 1
        elif action == "DOWN": head_y += 1
        elif action == "LEFT": head_x -= 1
        elif action == "RIGHT": head_x += 1

        new_head = (head_x, head_y)

        # Rand
        if head_x < 0 or head_x >= self.width or head_y < 0 or head_y >= self.height:
            return self.get_state(), -1.0, True

        # schlechter Block
        if new_head == self.bad_block:
            return self.get_state(), -1.0, True

        # Essen
        if new_head == self.food:
            self.snake.insert(0, new_head)
            self.score += 1
            self.food = self._random_empty_cell(exclude={self.bad_block})
            self.bad_block = self._random_empty_cell(exclude={self.food})
            return self.get_state(), +1.0, False

        # Normale Bewegung
        old_head = self.snake[0]
        self.snake.insert(0, new_head)
        self.snake.pop()

        # Reward shaping: näher zum Essen?
        old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        if new_dist < old_dist:
            return self.get_state(), +0.1, False
        elif new_dist > old_dist:
            return self.get_state(), -0.1, False
        else:
            return self.get_state(), 0.0, False

    def render(self):
        os.system("cls" if os.name == "nt" else "clear")
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if (x, y) == self.snake[0]:
                    row += "H "
                elif (x, y) in self.snake:
                    row += "S "
                elif (x, y) == self.food:
                    row += "F "
                elif (x, y) == self.bad_block:
                    row += "X "
                else:
                    row += ". "
            print(row)
        print("Score:", self.score)

class QLearningAgent:
    def __init__(self, actions, alpha=0.2, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_vals = self.Q[state]
        return max(q_vals, key=q_vals.get)

    def update(self, state, action, reward, next_state, done):
        q_current = self.Q[state][action]
        if done:
            target = reward
        else:
            next_best = max(self.Q[next_state].values())
            target = reward + self.gamma * next_best
        self.Q[state][action] = q_current + self.alpha * (target - q_current)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def run_episode(env, agent, visualize=False, delay=0.1):
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if visualize:
            env.render()
            time.sleep(delay)
    return total_reward, env.score

def save_agent(agent, path="snake_agent.pkl"):
    data = {
        "Q": dict(agent.Q),
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "epsilon": agent.epsilon,
        "epsilon_min": agent.epsilon_min,
        "epsilon_decay": agent.epsilon_decay,
        "actions": agent.actions
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_agent(path="snake_agent.pkl"):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return QLearningAgent(ACTIONS)
    with open(path, "rb") as f:
        data = pickle.load(f)
    agent = QLearningAgent(
        actions=data["actions"],
        alpha=data["alpha"],
        gamma=data["gamma"],
        epsilon=data["epsilon"],
        epsilon_min=data["epsilon_min"],
        epsilon_decay=data["epsilon_decay"]
    )
    agent.Q = defaultdict(lambda: {a: 0.0 for a in agent.actions})
    for state, qvals in data["Q"].items():
        agent.Q[state] = qvals
    return agent

if __name__ == "__main__":
    episodes = int(input("Wie viele Simulationen? "))
    vis_choice = input("Visualisierung? (J/N) ").strip().upper()
    visualize = (vis_choice == "J")

    agent = load_agent("snake_agent.pkl")
    env = SnakeGame()
    total_score_all = 0

    for ep in range(1, episodes + 1):
        total_reward, score = run_episode(env, agent, visualize=visualize, delay=0.1)
        agent.decay_epsilon()
        total_score_all += score
        if not visualize:
            print(f"Episode {ep}/{episodes} | Score={score} | Reward={total_reward:.2f} | Epsilon={agent.epsilon:.3f} | GesamtScore={total_score_all}")

    save_agent(agent)
    print("Agent gespeichert in snake_agent.pkl")