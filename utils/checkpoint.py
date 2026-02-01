import os
import torch


def save_model(model, save_dir, step=None, filename=None):
    """
    Save only model parameters (most common).

    If `step` is provided, saves as `model_step_{step}.pt`.
    Otherwise saves as `best.pt` (or `filename` if provided).
    """
    os.makedirs(save_dir, exist_ok=True)
    if filename is not None:
        save_path = os.path.join(save_dir, filename)
    elif step is None:
        save_path = os.path.join(save_dir, "best.pt")
    else:
        save_path = os.path.join(save_dir, f"model_step_{step}.pt")

    torch.save(model.state_dict(), save_path)
    print(f"[Checkpoint] Model saved to {save_path}")

def save_checkpoint(model, optimizer, step, save_dir):
    """
    保存模型 + optimizer + step
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ckpt_step_{step}.pt")

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path
    )

    print(f"[Checkpoint] Saved checkpoint to {save_path}")


def load_checkpoint(model, optimizer, ckpt_path):
    """
    恢复训练
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]

    print(f"[Checkpoint] Loaded checkpoint from {ckpt_path}, step={step}")
    return step
