import os

def save_train_plot(loss_history, reward_history, model_name, seed, game):
    import matplotlib.pyplot as plt

    os.makedirs("outputs/plots/train", exist_ok=True)
    fig_path = os.path.join("outputs/plots/train", f"{game}_{model_name}_seed{seed}_train.png")
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes[0].plot(loss_history)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Loss")
    axes[1].plot(reward_history)
    axes[1].set_title("Running Score")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
