import argparse
import json
import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim


# Upright view - no geometric transformation
class IdentityView:
    name = "identity"

    def apply(self, x: torch.Tensor) -> torch.Tensor: 
        return x
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor: # no transformation
        return x

# Rotate the image 180 degrees - flip - geometric transformation
class Rotation180View:
    name = "rot180"

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=2, dims=[2, 3]) #90*2 = 180
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=2, dims=[2, 3])   


#Vertical flip - upside down - geometric transformation
class VerticalFlipView:
    name = "vflip"

    #dim = batch_size, channel, height, width
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[2])
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[2])

# Horizontal flip - mirror - geometric transformation
class HorizontalFlipView:
    name = "hflip"

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[3])
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[3])


VIEW_MAP = {
    "identity":  IdentityView(),
    "rot180":    Rotation180View(),
    "vflip":     VerticalFlipView(),
    "hflip":     HorizontalFlipView(),
}

# convert jpg images to tensors and normalize
def load_image(path: str, size: int = 256) -> torch.Tensor:
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    return loader(Image.open(path).convert("RGB")).unsqueeze(0)

# laplacian to simplify image to separate high and low freq
# connection to paper: this is similar to the multi-scale nature of the diffusion model's noise estimation at different scales
def get_laplacian_pyramid(img: torch.Tensor, levels: int=4) -> List[torch.Tensor]:
    pyramid: List[torch.Tensor] = []
    current = img

    for i in range(levels):
        low = F.avg_pool2d(current, kernel_size=3, stride=1, padding=1)
        pyramid.append(current - low) # high-freq
        current = F.interpolate(low, scale_factor=0.5, mode="bilinear", align_corners=False)

    pyramid.append(current) # low-freq
    return pyramid

# Get sobel edge for each tensor for structural boundaries
# connection to paper: this is similar to the text-to-image diffusion model's semantic knowledge
def get_sobel_edges(img: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
    ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
    
    # Repeat kernels for all 3 channels (R,G,B) and then do grouped convolution
    kx = kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ex = F.conv2d(img, kx, padding=1, groups=3)
    ey = F.conv2d(img, ky, padding=1, groups=3)

    return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6) #return mag of gradient

# Loss function - combine 3 terms: laplacian pyramid MSE, AGB, sobel edge consistency
# connection to paper: gradients from both views are combined before each param update. similar to multi-view parallel desnoising update in paper.
def compute_loss(anagram: torch.Tensor, t_a: torch.Tensor, t_b: torch.Tensor, view_b, lambda_edge: float=0.5,lambda_lap: float=1.0) -> Tuple[torch.Tensor, float, float]:
   
    # 1. laplacian pyramid mse
    pyr_curr = get_laplacian_pyramid(anagram)
    pyr_a = get_laplacian_pyramid(t_a)
    pyr_b = get_laplacian_pyramid(t_b)
    lap_loss = torch.tensor(0.0, requires_grad=True)

    for j, (layer, la, lb) in enumerate(zip(pyr_curr, pyr_a, pyr_b)):
        l_a = F.mse_loss(layer, la)
        l_b = F.mse_loss(view_b.apply(layer), lb)

        # 2. AGB: boost whichever view is currently losing
        ratio = (l_b / (l_a + 1e-8)).detach().clamp(0.5, 2.0)
        lap_loss = lap_loss + (l_a + ratio * l_b) * (1.0 / (2**j))

    # sobel edge consistency
    edge_curr = get_sobel_edges(anagram)
    edge_a = get_sobel_edges(t_a)
    edge_b = get_sobel_edges(t_b)
    edge_loss = (F.mse_loss(edge_curr, edge_a) + F.mse_loss(view_b.apply(edge_curr), edge_b))

    total_loss = (lambda_lap * lap_loss) + (lambda_edge * edge_loss) # laplacian + edge consistency

    return total_loss, lap_loss.item(), edge_loss.item()

# Tiny reproduction of paper - what's different - similar to algo 1 of the paper - mathematical core of parallel updates without GPU support
def run_anagram(t_a: torch.Tensor, t_b: torch.Tensor, view_b, steps: int=1000, lr: float=0.01, verbose: bool=True) -> Tuple[torch.Tensor, List[dict]]:
    
    """
    Steps to follow:
    1. for every view, transform the current image and computer gradient estimate
        in our project, we use the MSE gradient, the actual paper uses diffusion score/noise estimate
    2. Inverse-transform each of the gradients back to the pixel space
    3. Average gradients across views before the parameter update.
    """

    # choose middle to be safe starting point, optimize till convergence
    anagram   = nn.Parameter(torch.rand_like(t_a) * 0.5 + 0.25)
    optimizer = optim.Adam([anagram], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    history: List[dict] = []

    # for all steps, compute loss and update anagram. Save history for vis and evals.
    for i in range(steps + 1):
        optimizer.zero_grad()
        total_loss, lap_l, edge_l = compute_loss(anagram, t_a, t_b, view_b)
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            anagram.clamp_(0.0, 1.0)

        if i % 100 == 0:
            with torch.no_grad():
                mse_a = F.mse_loss(anagram, t_a).item()
                mse_b = F.mse_loss(view_b.apply(anagram), t_b).item()

            history.append({"step": i, "total_loss": total_loss.item(), "lap_loss": lap_l, "edge_loss": edge_l, "mse_a": mse_a, "mse_b": mse_b})

            if verbose:
                print(f"  step {i:5d} | loss {total_loss.item():.3f} | lap {lap_l:.3f} | edge {edge_l:.3f} | mse_A {mse_a:.3f} | mse_B {mse_b:.3f}")

    return anagram.detach(), history


# baseline - average pixels of both images (one upright, one inversed) to get target
def naive_baseline(t_a: torch.Tensor, t_b: torch.Tensor, view_b) -> torch.Tensor:
    t_b_in_a_space = view_b.inverse(t_b)
    return ((t_a + t_b_in_a_space)/2.0).clamp(0.0, 1.0)

# calculate structural similarity index (SSIM) bw two images. Better perpetual sim that MSE.
def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(ssim(a, b, data_range=1.0, channel_axis=2))

# convert tensor to numpy for evals and visualisation
def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

# MSE, SSIM, balance of MSEs (MSE_A / MSE_B)    
# MSE low means closer to combined image, high means farther away
# SSIM low means less structurally similar, high means more
# Balance means both images contribute how much to combined. 1 means equal, less than 1 means A dominates, more than 1 means B dominates
def evaluate(anagram: torch.Tensor, t_a: torch.Tensor, t_b: torch.Tensor, view_b, label: str = "") -> dict:
    with torch.no_grad():
        view_a_img = anagram
        view_b_img = view_b.apply(anagram)

        # mses of both
        mse_a = F.mse_loss(view_a_img, t_a).item()
        mse_b = F.mse_loss(view_b_img, t_b).item()

        # ssims of both
        ssim_a = compute_ssim(to_numpy(view_a_img), to_numpy(t_a))
        ssim_b = compute_ssim(to_numpy(view_b_img), to_numpy(t_b))

    return {"label": label, "mse_view_a": round(mse_a, 3), "mse_view_b": round(mse_b, 3), "ssim_view_a": round(ssim_a, 3), "ssim_view_b": round(ssim_b, 3), "balance_ratio": round(mse_a / (mse_b + 1e-8), 3)}

# og images, our tiny reproduction, and naive baseline to compare
def save_comparison_figure(anagram, naive, t_a, t_b, view_b, m_opt, m_naive, view_name: str, out_path: str) -> None:
    fig = plt.figure(figsize=(13, 6))
    fig.suptitle(f"Visual Anagram Tiny Reproduction - View: {view_name}") #, fontsize=13, fontweight="bold", y=1.01,)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.25)

    def show(ax, tensor, title, sub=""):
        ax.imshow(to_numpy(tensor))
        ax.set_title(title, fontsize=10, fontweight="bold")

        if sub:
            ax.set_xlabel(sub, fontsize=8, color="#555555")

        ax.axis("off")

    # Column 0 - og images
    show(fig.add_subplot(gs[0, 0]), t_a, "Target A\n(upright)")
    show(fig.add_subplot(gs[1, 0]), t_b, f"Target B\n({view_name})")

    # Column 1 - tiny reproduction
    show(fig.add_subplot(gs[0, 1]), anagram, "Tiny Reproduction - upright view", f"MSE {m_opt['mse_view_a']:.3f}  SSIM {m_opt['ssim_view_a']:.3f}")
    show(fig.add_subplot(gs[1, 1]), view_b.apply(anagram), f"Tiny Reproduction - {view_name} view", f"MSE {m_opt['mse_view_b']:.3f}  SSIM {m_opt['ssim_view_b']:.3f}")

    # Column 2 — naive baseline
    show(fig.add_subplot(gs[0, 2]), naive, "Naive blend - upright", f"MSE {m_naive['mse_view_a']:.3f}  SSIM {m_naive['ssim_view_a']:.3f}")
    show(fig.add_subplot(gs[1, 2]), view_b.apply(naive), f"Naive blend - {view_name}", f"MSE {m_naive['mse_view_b']:.3f}  SSIM {m_naive['ssim_view_b']:.3f}")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Results in: {out_path}")

# total loss curve + per-view MSE curves to show convergence and balance of optimization
def save_loss_curves(history: List[dict], out_path: str) -> None:
    steps = []
    total = []
    mse_a = []
    mse_b = []

    for h in history:
        steps.append(h["step"])
        total.append(h["total_loss"])
        mse_a.append(h["mse_a"])
        mse_b.append(h["mse_b"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Optimisation Convergence", fontsize=12, fontweight="bold")

    axes[0].plot(steps, total, color="#1a73e8", linewidth=2)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, mse_a, color="#34a853", linewidth=2, label="View A (upright)")
    axes[1].plot(steps, mse_b, color="#ea4335", linewidth=2, label="View B (transformed)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Per-View MSE  (balance check)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f" Figures saved in: {out_path}")

# generate table
def save_metrics_table(all_metrics: List[dict], out_path: str) -> None:
    headers = ["Method", "MSE (A)", "MSE (B)", "SSIM (A)", "SSIM (B)", "Balance"]

    rows = []
    for i in all_metrics:
        row = [i["label"], i["mse_view_a"], i["mse_view_b"], i["ssim_view_a"], i["ssim_view_b"], i["balance_ratio"]]
        rows.append(row)

    fig, ax = plt.subplots(figsize=(11, max(2.5, len(rows) * 0.65 + 1.2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    header_color = "#2c3e50"
    for j in range(len(headers)):
        tbl[0, j].set_facecolor(header_color)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        bg = "white"
        for j in range(len(headers)):
            tbl[i, j].set_facecolor(bg)

    tbl.auto_set_column_width(col=0)

    ax.set_title("Quantitative Comparison: Tiny Reproduction vs. Naive Baseline", fontsize=12, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Results saved in: {out_path}")

# Runs all the experiments and creates the respective folders to save the experiments in
# Structure: results/{name_a}_{name_b}/{transform_data}
def run_all_experiments(image_pairs: List[Tuple[str, str]], views: list, steps: int = 1000, out_dir: str = "results") -> List[dict]:
    os.makedirs(out_dir, exist_ok=True)
    all_metrics: List[dict] = []

    # image folders
    for path_a, path_b in image_pairs:
        name_a = os.path.splitext(os.path.basename(path_a))[0]
        name_b = os.path.splitext(os.path.basename(path_b))[0]
        
        # Create subfolder like 'results/img1_img2/' for this pair of images
        pair_folder_name = f"{name_a}_{name_b}"
        pair_dir = os.path.join(out_dir, pair_folder_name)
        os.makedirs(pair_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Pairs: {pair_dir}")
        print(f"{'='*60}")

        t_a = load_image(path_a)
        t_b = load_image(path_b)

        # define the view
        for view in views:
            exp_label = f"{view.name}_{name_a}_{name_b}"
            print(f"\nRunning View: {view.name}")

            # run anagram and the naive baseline to compare with ours
            anagram, history = run_anagram(t_a, t_b, view, steps=steps)
            naive = naive_baseline(t_a, t_b, view)

            # evaluate tiny reproduction with the naive baseline
            m_opt = evaluate(anagram, t_a, t_b, view, label=f"Ours ({exp_label})")
            m_naive = evaluate(naive, t_a, t_b, view, label=f"Naive ({exp_label})")
            all_metrics.extend([m_opt, m_naive])

            # save in files
            base = os.path.join(pair_dir, view.name)
            
            plt.imsave(f"{base}_normal.png", to_numpy(anagram))
            plt.imsave(f"{base}_transformed.png", to_numpy(view.apply(anagram)))
            
            save_comparison_figure(anagram, naive, t_a, t_b, view, m_opt, m_naive, view.name, f"{base}_comparison.png")
            save_loss_curves(history, f"{base}_loss_curves.png")

    # create table
    save_metrics_table(all_metrics, os.path.join(out_dir, "summary_metrics.png"))
    
    # save all metrics in one json file
    with open(os.path.join(out_dir, "all_data.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics

def main():
    parser = argparse.ArgumentParser(description="Visual Anagrams - Tiny Reproduction Track 1")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--views", nargs="+", choices=list(VIEW_MAP.keys()), default=["rot180", "vflip", "hflip"]) #add all the transormation to want to do here
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    selected_views = []
    for view_name in args.views:
        view_transform = VIEW_MAP[view_name]
        selected_views.append(view_transform)

    # images you want to compare or transform
    image_pairs = [
        # ("face.png", "mountain.png"),
        ("img1.jpg", "img2.jpg"),
        ("giraffe.jpg", "penguin.jpg"),
    ]

    run_all_experiments(image_pairs, selected_views, args.steps, args.out_dir)

if __name__ == "__main__":
    main()