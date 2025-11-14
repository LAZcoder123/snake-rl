# 🐍 Snake KI mit Q-Learning

Dies ist ein einfaches Reinforcement-Learning-Projekt, bei dem eine KI lernt, das Spiel Snake zu spielen.  
Die KI verwendet Q-Learning mit einer ASCII-Visualisierung im Terminal.

## 🎮 Spielregeln

- Die Schlange bewegt sich auf einem 10×10-Feld.
- Essen (F) bringt +1 Punkt und lässt die Schlange wachsen.
- Schlechte Blöcke (X) oder Wände führen zum Tod (–1 Punkt).
- Kleine Belohnungen für Bewegung in Richtung Essen (+0.1) oder weg davon (–0.1).

## 🧠 KI-Details

- Q-Learning mit Epsilon-Greedy-Strategie
- Zustände sind relativ zur Position des Essens und des schlechten Blocks
- Fortschritt wird in `snake_agent.pkl` gespeichert

## 🛠️ Nutzung

```bash
python snake_rl.py
