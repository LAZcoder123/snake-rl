# snake_rl.py
import os
import sys
import random
import time
import pickle
from collections import defaultdict

# Grid und Aktionen
WIDTH, HEIGHT = 10, 10
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Speicherpfad: immer im Ordner dieser Datei
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
SAVE_PATH = os.path.join(BASE_DIR, "snake_agent.pkl")

# Logging/Debug
LOG_INTERVAL = 100
DEBUG_INTERVAL_STEPS = 1000
DEBUG_INTERVAL_SECS = 5
MAX_STEPS = 20000

# Visualisierung
RENDER_DELAY = 0.05          # Sekunden zwischen Frames
RENDER_EVERY_STEPS = 1       # nur jedes N-te Frame rendern (gegen Flackern)

# Stall-Erkennung (Hänger vermeiden)
STALL_HEAD_STEPS = 300       # wenn Kopf sich so viele Schritte nicht bewegt -> Stall
STALL_NO_IMPROVE_SECS = 15   # wenn Score/Distanz sich so lange nicht verbessert -> Stall

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
        # Sicherheit: keine Überschneidungen
        if self.bad_block == self.food:
            self.bad_block = self._random_empty_cell(exclude={self.food})
        return self.get_state()

    def _random_empty_cell(self, exclude=None):
        exclude = set(exclude) if exclude else set()
        exclude.update(self.snake)
        attempts = 0
        while attempts < self.width * self.height * 4:
            attempts += 1
            cell = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if cell not in exclude:
                return cell
        # Notfall: suche systematisch
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in exclude:
                    return (x, y)
        # Fallback (sollte nie passieren)
        return (0, 0)

    def get_state(self):
        hx, hy = self.snake[0]
        fx, fy = self.food
        bx, by = self.bad_block
        def rel(a, b): return (a > b) - (a < b)
        return (rel(fx, hx), rel(fy, hy), rel(bx, hx), rel(by, hy))

    def step(self, action):
        hx, hy = self.snake[0]
        if action == "UP":    hy -= 1
        elif action == "DOWN": hy += 1
        elif action == "LEFT": hx -= 1
        elif action == "RIGHT":hx += 1
        new_head = (hx, hy)

        # Randkollision
        if hx < 0 or hx >= self.width or hy < 0 or hy >= self.height:
            return self.get_state(), -1.0, True

        # Schlechter Block
        if new_head == self.bad_block:
            return self.get_state(), -1.0, True

        # Essen
        if new_head == self.food:
            self.snake.insert(0, new_head)
            self.score += 1
            # Neues Food/Bad-Block ohne Überschneidung
            self.food = self._random_empty_cell(exclude={self.bad_block}.union(self.snake))
            self.bad_block = self._random_empty_cell(exclude={self.food}.union(self.snake))
            return self.get_state(), +1.0, False

        # Normale Bewegung
        old_head = self.snake[0]
        self.snake.insert(0, new_head)
        self.snake.pop()

        # Reward shaping: Manhattan-Distanz zum Food
        old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        reward = +0.1 if new_dist < old_dist else -0.1 if new_dist > old_dist else 0.0
        return self.get_state(), reward, False

    def render(self):
        # ASCII-Render
        os.system("cls" if os.name == "nt" else "clear")
        head = self.snake[0]
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) == head:
                    row.append("H")
                elif (x, y) in self.snake:
                    row.append("S")
                elif (x, y) == self.food:
                    row.append("F")
                elif (x, y) == self.bad_block:
                    row.append("X")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        print("\n".join(lines))
        print(f"Score: {self.score}")
        sys.stdout.flush()

class QLearningAgent:
    def __init__(self, actions, alpha=0.2, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
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
        qvals = self.Q[state]
        return max(qvals, key=qvals.get)

    def update(self, state, action, reward, next_state, done):
        q = self.Q[state][action]
        target = reward if done else reward + self.gamma * max(self.Q[next_state].values())
        self.Q[state][action] = q + self.alpha * (target - q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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

def run_episode(env, agent, ep_num, visualize=False):
    state = env.reset()
    total_reward = 0.0
    done = False
    steps = 0

    # Debug/Monitoring
    last_time_debug = time.time()
    last_improve_time = time.time()
    last_head = env.snake[0]
    head_same_steps = 0
    last_dist = abs(last_head[0] - env.food[0]) + abs(last_head[1] - env.food[1])

    safe_print(f"[Episode {ep_num}] gestartet.")

    while not done and steps < MAX_STEPS:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1

        # Stall-Erkennung: Kopfbewegung
        head = env.snake[0]
        if head == last_head:
            head_same_steps += 1
        else:
            head_same_steps = 0
            last_head = head

        # Stall-Erkennung: Verbesserung (Score oder Distanz)
        dist = abs(head[0] - env.food[0]) + abs(head[1] - env.food[1])
        if env.score > 0 or dist < last_dist:
            last_improve_time = time.time()
            last_dist = dist

        # Zeit-/Schrittbasierte Debug-Ausgaben
        now = time.time()
        if steps % DEBUG_INTERVAL_STEPS == 0:
            safe_print(f"[Episode {ep_num}] Schritte={steps} | Score={env.score} | Kopf={head} | Distanz={dist}")
        if now - last_time_debug >= DEBUG_INTERVAL_SECS:
            safe_print(f"[Episode {ep_num}] Schritte={steps} | Score={env.score} | Kopf={head} | Distanz={dist}")
            last_time_debug = now

        # Visualisierung (gedrosselt)
        if visualize and (steps % RENDER_EVERY_STEPS == 0):
            env.render()
            time.sleep(RENDER_DELAY)

        # Stall-Abbruchbedingungen
        if head_same_steps >= STALL_HEAD_STEPS:
            safe_print(f"[Episode {ep_num}] Stall erkannt: Kopf {head_same_steps} Schritte unverändert. Breche Episode ab.")
            done = True
        if time.time() - last_improve_time >= STALL_NO_IMPROVE_SECS:
            safe_print(f"[Episode {ep_num}] Keine Verbesserung seit {STALL_NO_IMPROVE_SECS}s. Breche Episode ab.")
            done = True

    if steps >= MAX_STEPS and not done:
        safe_print(f"[Episode {ep_num}] abgebrochen nach MAX_STEPS={MAX_STEPS}.")
        done = True

    safe_print(f"[Episode {ep_num}] beendet: Score={env.score} | Schritte={steps} | TotalReward={total_reward:.2f}")
    return total_reward, env.score, steps

def main():
    safe_print("Snake RL startet...")

    try:
        episodes = int(input("Wie viele Episoden? "))
    except Exception:
        episodes = 100
        safe_print("Ungültige Eingabe, setze Episoden=100.")

    visualize = input("Visualisierung aktivieren? (J/N): ").strip().upper() == "J"

    agent = load_agent(SAVE_PATH)
    env = SnakeGame()
    total_score_all = 0
    total_steps_all = 0
    start_time_all = time.time()

    try:
        for ep in range(1, episodes + 1):
            reward, score, steps = run_episode(env, agent, ep, visualize=visualize)
            agent.decay_epsilon()
            total_score_all += score
            total_steps_all += steps

            # Episoden-Log (immer)
            safe_print(f"[Episode {ep}/{episodes}] Score={score} | GesamtScore={total_score_all} | Schritte={steps} | Epsilon={agent.epsilon:.3f} | Zustände={len(agent.Q)}")

            # Periodisches Gesamt-Log
            if ep % LOG_INTERVAL == 0:
                safe_print(f"--- LOG --- Episoden={ep} | GesamtScore={total_score_all} | GesamtSchritte={total_steps_all} | Epsilon={agent.epsilon:.3f} | Zustände={len(agent.Q)}")

    except KeyboardInterrupt:
        safe_print("\nAbgebrochen. Speichere aktuellen Agent-Stand...")
        save_agent(agent, SAVE_PATH)
        safe_print(f"Agent gespeichert in: {SAVE_PATH}")
        return

    # Abschluss
    save_agent(agent, SAVE_PATH)
    elapsed = int(time.time() - start_time_all)
    safe_print(f"Agent gespeichert in: {SAVE_PATH}")
    safe_print(f"GesamtScore nach {episodes} Episoden: {total_score_all}")
    safe_print(f"GesamtSchritte: {total_steps_all} | Dauer: {elapsed}s")
    safe_print(f"Gelernte Zustände: {len(agent.Q)}")

if __name__ == "__main__":
    main()