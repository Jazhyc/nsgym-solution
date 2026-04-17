import torch
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from AAMAS_Comp.optimizers.cbp import ContinualBackpropagation

# ── Model ────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim=20, hidden=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Synthetic continual-learning tasks ───────────────────────────────────────

def make_task(n=1000, in_dim=20, seed=0):
    """Binary classification task: random linear boundary."""
    torch.manual_seed(seed)
    X = torch.randn(n, in_dim)
    w = torch.randn(in_dim)
    y = (X @ w > 0).long()
    return TensorDataset(X, y)


# ── Training helpers ─────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, cbp=None, global_step=0):
    model.train()
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        if cbp:
            cbp.step(global_step)
        global_step += 1
        total_loss += loss.item()
    return total_loss / len(loader), global_step


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / total


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(label, make_optimizer_fn, use_cbp=False,
                   epochs_per_task=30, seed=42):
    torch.manual_seed(seed)
    model = MLP()
    optimizer = make_optimizer_fn(model)
    cbp = ContinualBackpropagation(model, optimizer=optimizer,
                                   reset_rate=0.02, maturity_threshold=30) if use_cbp else None

    task_a_train = DataLoader(make_task(800, seed=0),  batch_size=64, shuffle=True)
    task_a_test  = DataLoader(make_task(200, seed=1),  batch_size=200)
    task_b_train = DataLoader(make_task(800, seed=10), batch_size=64, shuffle=True)
    task_b_test  = DataLoader(make_task(200, seed=11), batch_size=200)

    history = {'task_a': [], 'task_b': [], 'phase': []}
    step = 0

    # Phase 1 — train on Task A
    for e in range(epochs_per_task):
        _, step = train_epoch(model, task_a_train, optimizer, cbp, step)
        history['task_a'].append(evaluate(model, task_a_test))
        history['task_b'].append(evaluate(model, task_b_test))
        history['phase'].append('A')

    # Phase 2 — train on Task B (forgetting test)
    for e in range(epochs_per_task):
        _, step = train_epoch(model, task_b_train, optimizer, cbp, step)
        history['task_a'].append(evaluate(model, task_a_test))
        history['task_b'].append(evaluate(model, task_b_test))
        history['phase'].append('B')

    return history


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f0f0f')
    fig.suptitle('Continual Learning: CBP vs Adam vs SGD\n'
                 'Task A trained first, then Task B — measures catastrophic forgetting',
                 color='white', fontsize=13, y=1.01)

    colors = {
        'Adam + CBP': '#00e5ff',
        'Adam':       '#ff6b6b',
        'SGD':        '#a8ff78',
    }

    for ax, task_key, task_label in zip(axes, ['task_a', 'task_b'],
                                        ['Task A accuracy (forgetting test)', 'Task B accuracy']):
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#aaa')
        ax.spines[:].set_color('#333')
        ax.set_title(task_label, color='white', fontsize=11)
        ax.set_xlabel('Epoch', color='#aaa')
        ax.set_ylabel('Accuracy', color='#aaa')
        ax.axvline(x=30, color='#555', linestyle='--', linewidth=1.2, label='Task switch')
        ax.set_ylim(0.4, 1.02)

        for label, hist in results.items():
            ax.plot(hist[task_key], label=label, color=colors[label], linewidth=2, alpha=0.9)

        ax.legend(facecolor='#111', labelcolor='white', framealpha=0.7)

    plt.tight_layout()
    plt.savefig('./cbp_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='#0f0f0f')
    print("Plot saved → cbp_comparison.png")
    plt.show()


if __name__ == '__main__':
    print("Running Adam + CBP ...")
    r_cbp  = run_experiment('Adam + CBP', lambda m: torch.optim.Adam(m.parameters(), lr=1e-3), use_cbp=True)

    print("Running Adam ...")
    r_adam = run_experiment('Adam',       lambda m: torch.optim.Adam(m.parameters(), lr=1e-3), use_cbp=False)

    print("Running SGD ...")
    r_sgd  = run_experiment('SGD',        lambda m: torch.optim.SGD(m.parameters(),  lr=0.01, momentum=0.9), use_cbp=False)

    results = {'Adam + CBP': r_cbp, 'Adam': r_adam, 'SGD': r_sgd}

    # Print summary table
    print("\n── Final accuracy summary ──────────────────────────────")
    print(f"{'Method':<14} {'Task A (after B train)':>22}  {'Task B (final)':>16}")
    print("-" * 56)
    for label, hist in results.items():
        a_final = hist['task_a'][-1]
        b_final = hist['task_b'][-1]
        print(f"{label:<14} {a_final:>22.3f}  {b_final:>16.3f}")

    plot_results(results)
