import os
import sys
import random
import time
import pickle
from collections import defaultdict

# Grid und Aktionen
WIDTH, HEIGHT = 10, 10
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Speicherpfad: immer im gleichen Ordner wie snake_rl.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
SAVE_PATH = os.path.join(BASE_DIR, "snake_agent.pkl")

# Logging/Debug-Parameter
LOG_INTERVAL = 100          # Episoden-Zusammenfassung alle X Episoden
DEBUG_INTERVAL_STEPS = 1000 # Live-Ticker: Ausgabe alle X Schritte innerhalb einer Episode
DEBUG_INTERVAL_SECS = 5     # Live-Ticker: Ausgabe alle Y Sekunden innerhalb einer Episode
MAX_STEPS = 20000           # Harte Obergrenze pro Episode

def safe_print(msg):
    print(msg)
    sys.stdout.flush()

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
        # Robust: falls das Grid voll wäre, brich nach vielen Versuchen ab
        attempts = 0
        while True:
            attempts += 1
            if attempts > self.width * self.height * 4:
                return (0, 0)
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

        # Randkollision
        if head_x < 0 or head_x >= self.width or head_y < 0 or head_y >= self.height:
            return self.get_state(), -1.0, True

        # Schlechter Block
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

def run_episode(env, agent, visualize=False, delay=0.05):
    state = env.reset()
    total_reward = 0.0
    done = False
    steps = 0
    last_time_debug = time.time()

    safe_print("Episode gestartet...")

    while not done and steps < MAX_STEPS:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1

        # Schrittbasierter Live-Ticker
        if not visualize and steps % DEBUG_INTERVAL_STEPS == 0:
            safe_print(f"Episode läuft noch... {steps} Schritte bisher")

        # Zeitbasierter Live-Ticker
        now = time.time()
        if not visualize and now - last_time_debug >= DEBUG_INTERVAL_SECS:
            safe_print(f"Episode läuft noch... {steps} Schritte, {int(now - last_time_debug)}s seit letzter Meldung")
            last_time_debug = now

        if visualize:
            env.render()
            time.sleep(delay)

    if steps >= MAX_STEPS and not done:
        safe_print(f"Episode abgebrochen nach MAX_STEPS={MAX_STEPS}.")
        done = True

    return total_reward, env.score, steps

def save_agent(agent, path=SAVE_PATH):
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

def load_agent(path=SAVE_PATH):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        safe_print("Kein bestehender Agent gefunden. Starte neu.")
        return QLearningAgent(ACTIONS)
    with open(path, "rb") as f:
        data = pickle.load(f)
    agent = QLearningAgent(
        actions=data.get("actions", ACTIONS),
        alpha=data.get("alpha", 0.2),
        gamma=data.get("gamma", 0.95),
        epsilon=data.get("epsilon", 1.0),
        epsilon_min=data.get("epsilon_min", 0.05),
        epsilon_decay=data.get("epsilon_decay", 0.995)
    )
    agent.Q = defaultdict(lambda: {a: 0.0 for a in agent.actions})
    for state, qvals in data.get("Q", {}).items():
        agent.Q[state] = qvals
    safe_print(f"Agent geladen aus: {path}")
    return agent

def main():
    safe_print("Snake RL startet... Bitte Eingaben machen:")
    try:
        raw = input("Wie viele Simulationen? (Enter=100) ").strip()
        episodes = int(raw) if raw else 100
    except Exception:
        episodes = 100
        safe_print("Hinweis: Ungültige Eingabe, setze Episoden=100.")

    vis_choice = input("Visualisierung? (J/N, Enter=N) ").strip().upper()
    visualize = (vis_choice == "J")

    if visualize:
        safe_print("Visualisierung aktiv. Konsole wird regelmäßig neu gezeichnet.")

    agent = load_agent(SAVE_PATH)
    env = SnakeGame()
    total_score_all = 0
    total_steps_all = 0
    start_time_all = time.time()

    try:
        for ep in range(1, episodes + 1):
            safe_print(f"Starte Episode {ep}/{episodes} ...")
            total_reward, score, steps = run_episode(env, agent, visualize=visualize, delay=0.05)
            agent.decay_epsilon()
            total_score_all += score
            total_steps_all += steps

            if not visualize and ep % LOG_INTERVAL == 0:
                safe_print(
                    f"Episode {ep}/{episodes} | letzter Score={score} | GesamtScore={total_score_all} "
                    f"| Schritte in Episode={steps} | GesamtSchritte={total_steps_all} "
                    f"| Epsilon={agent.epsilon:.3f} | Zustände gelernt={len(agent.Q)}"
                )

    except KeyboardInterrupt:
        safe_print("\nAbgebrochen. Speichere aktuellen Agent-Stand...")
        save_agent(agent, SAVE_PATH)
        safe_print(f"Agent gespeichert in: {SAVE_PATH}")
        return

    # Abschluss nach allen Episoden
    save_agent(agent, SAVE_PATH)
    elapsed = time.time() - start_time_all
    safe_print(f"Agent gespeichert in: {SAVE_PATH}")
    safe_print(f"GesamtScore nach {episodes} Episoden: {total_score_all}")
    safe_print(f"GesamtSchritte: {total_steps_all} | Dauer gesamt: {int(elapsed)}s")
    safe_print(f"Gelernte Zustände: {len(agent.Q)}")

if __name__ == "__main__":
    main()