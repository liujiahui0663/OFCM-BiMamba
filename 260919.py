import os
import glob
import random
import time
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


RGB_ROOT = r"<RGB图像序列根目录>"
FLOW_ROOT = r"<光流数据根目录>"
OUTPUT_DIR = "<输出目录>"
MODEL_TAG = "<模型标识>"

EARLY_WARNING_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), OUTPUT_DIR, "<提前量实验子目录>"
)
EARLY_WARNING_WINDOWS = "<提前量窗口配置列表>"
EVENT_FRAME_COUNT = "<单个事件的总帧数>"
HISTORY_WINDOW_FRAMES = "<历史窗口帧数>"

IMG_SIZE = "<输入图像尺寸>"
SEQ_LEN = "<输入序列长度>"
BATCH_SIZE = "<批次大小>"
EPOCHS = "<训练轮数>"
LR = "<学习率>"
WEIGHT_DECAY = "<权重衰减系数>"
NUM_WORKERS = "<数据加载线程数>"
DEVICE = "<运行设备>"
NUM_RUNS = "<独立实验次数>"
SEEDS = "<随机种子列表>"

FEATURE_DIM = "<特征维度>"
ATTENTION_HEADS = "<注意力头数>"
FEATURE_MAP_SIZE = "<特征图空间尺寸>"
SSM_DT_RANK = "<状态空间模型秩>"
TEMPORAL_KERNEL_SIZE = "<时序卷积核大小>"
TEMPORAL_PADDING = "<时序卷积填充大小>"
MOTION_DROPOUT = "<运动模块丢弃率>"
CLASSIFIER_DROPOUT = "<分类头丢弃率>"
NUM_CLASSES = "<类别数量>"
RECENCY_INIT = "<近期先验初始强度>"

LABEL_SMOOTHING = "<标签平滑系数>"
HOLDOUT_RATIO = "<验证集与测试集总占比>"
TEST_RATIO_IN_HOLDOUT = "<留出数据中的测试集占比>"
BRIGHTNESS_JITTER = "<亮度增强幅度>"
CONTRAST_JITTER = "<对比度增强幅度>"
SATURATION_JITTER = "<饱和度增强幅度>"
HUE_JITTER = "<色调增强幅度>"
GRAD_CLIP_NORM = "<梯度裁剪阈值>"

PROFILE_WARMUP_STEPS = "<性能测试预热次数>"
PROFILE_ITERATIONS = "<性能测试迭代次数>"
CAM_RANDOM_SEED = "<可视化抽样随机种子>"
CAM_SAMPLE_COUNT = "<可视化样本数量>"
CAM_MAX_COLUMNS = "<可视化最大列数>"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LaneDataset(Dataset):
    def __init__(self, rgb_folders, flow_root, seq_len, transform=None,
                 window_start=0, window_end=None, return_metadata=False,
                 window_specs=None, assignment_seed=0):
        self.rgb_folders = rgb_folders
        self.flow_root = flow_root
        self.seq_len = seq_len
        self.transform = transform
        self.window_start = window_start
        self.window_end = window_end
        self.return_metadata = return_metadata
        self.window_specs = window_specs
        self.assignment_seed = assignment_seed
        self.active_window_ids = None
        if self.window_specs is not None:
            self.set_epoch(0)

    def __len__(self):
        return len(self.rgb_folders)

    def get_label(self, name):
        if name.startswith("L"):
            return 0
        elif name.startswith("R"):
            return 1
        elif name.startswith("K"):
            return 2
        else:
            raise ValueError(f"Unknown class: {name}")

    def set_epoch(self, epoch):

        if self.window_specs is None:
            return
        rng = np.random.default_rng(self.assignment_seed + epoch)
        self.active_window_ids = rng.integers(0, len(self.window_specs), size=len(self.rgb_folders))

    def _window_for_item(self, item_index):
        if self.window_specs is None:
            return self.window_start, self.window_end
        if self.active_window_ids is None:
            self.set_epoch(0)
        spec = self.window_specs[int(self.active_window_ids[item_index])]
        return spec["Start"], spec["End"]

    def temporal_sampling(self, frame_num, window_start=None, window_end=None):
        window_start = self.window_start if window_start is None else window_start
        if window_end is None:
            window_end = frame_num if self.window_end is None else self.window_end
        if window_start < 0 or window_end > frame_num or window_end <= window_start:
            raise ValueError(
                f"Invalid observation window [{window_start}:{window_end}] for {frame_num} RGB frames."
            )
        idx = np.linspace(window_start, window_end - 1, self.seq_len).astype(np.int32)
        return idx

    def __getitem__(self, idx):
        rgb_folder = self.rgb_folders[idx]
        video_name = os.path.basename(rgb_folder)
        label = self.get_label(video_name)

        img_list = sorted(glob.glob(os.path.join(rgb_folder, "*.jpg")))
        if len(img_list) == 0:
            img_list = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))

        window_start, window_end = self._window_for_item(idx)
        sample_idx = self.temporal_sampling(len(img_list), window_start, window_end)

        rgb_seq = []
        for i in sample_idx:
            img = Image.open(img_list[i]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            rgb_seq.append(img)
        rgb_seq = torch.stack(rgb_seq, dim=0)

        flow_path = os.path.join(self.flow_root, video_name + ".npy")
        flow_data = np.load(flow_path, mmap_mode='r')
        if sample_idx.max() >= flow_data.shape[0]:
            raise ValueError(
                f"Flow index {sample_idx.max()} exceeds available flow length {flow_data.shape[0]} for {video_name}."
            )
        flow_seq_np = np.array(flow_data[sample_idx])

        flow_seq = []
        for i in range(self.seq_len):
            f = torch.tensor(flow_seq_np[i], dtype=torch.float32)
            f = f.permute(2, 0, 1)
            f = F.interpolate(f.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(
                0)
            flow_seq.append(f)

        flow_seq = torch.stack(flow_seq, dim=0)

        if self.return_metadata:
            return rgb_seq, flow_seq, label, video_name
        return rgb_seq, flow_seq, label


class ResEncoder(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if in_channel == 2:
            net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
            net.conv1.weight.data = net.conv1.weight.data[:, :2, :, :].clone()
        self.encoder = nn.Sequential(*list(net.children())[:-2])

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.encoder(x)
        return feat


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=FEATURE_DIM, heads=ATTENTION_HEADS):
        super().__init__()

        self.flow_mask_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())


        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.pos_embed = nn.Parameter(torch.randn(1, dim, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE) * 0.02)


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.factor_fc = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())


        self.out_linear = nn.Linear(dim, dim)
        self.cam_target = nn.Identity()

    def forward(self, rgb_feat, flow_feat, B, T):
        BT, D, H, W = rgb_feat.shape

        flow_mask = self.flow_mask_conv(flow_feat)
        rgb_modulated = rgb_feat * flow_mask

        rgb_mod_pe = rgb_modulated + self.pos_embed
        flow_feat_pe = flow_feat + self.pos_embed

        rgb_flat = rgb_mod_pe.view(BT, D, -1).permute(0, 2, 1)
        flow_flat = flow_feat_pe.view(BT, D, -1).permute(0, 2, 1)

        flow_modulated, _ = self.attn(query=flow_flat, key=rgb_flat, value=rgb_flat)

        flow_pooled = self.gap(flow_feat).view(BT, D)
        factor = self.factor_fc(flow_pooled).unsqueeze(1)

        flow_original_flat = flow_feat.view(BT, D, -1).permute(0, 2, 1)
        final_flow_flat = flow_original_flat + factor * flow_modulated

        final_flow_spatial = final_flow_flat.permute(0, 2, 1).view(BT, D, H, W)
        final_flow_spatial = self.cam_target(final_flow_spatial)

        out_seq = self.gap(final_flow_spatial).view(BT, D)
        out_seq = self.out_linear(out_seq)

        out_seq = out_seq.view(B, T, D)
        return out_seq


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, dt_rank=SSM_DT_RANK):
        super().__init__()
        self.d_model = d_model
        self.x_proj = nn.Linear(d_model, dt_rank + d_model * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_model + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        B, T, D = x.shape
        x_proj = self.x_proj(x)
        delta, B_proj, C_proj = torch.split(x_proj, [x_proj.shape[-1] - D * 2, D, D], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())

        y = torch.zeros_like(x)
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        for t in range(T):
            delta_t, x_t, B_t, C_t = delta[:, t], x[:, t], B_proj[:, t], C_proj[:, t]
            dA = torch.exp(delta_t * A)
            dB = delta_t * B_t
            h = dA * h + dB * x_t
            y[:, t] = h * C_t
        return y + x * self.D


class MambaBlock(nn.Module):
    def __init__(self, dim=FEATURE_DIM):
        super().__init__()
        self.upper = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        self.conv = nn.Conv1d(dim, dim, kernel_size=TEMPORAL_KERNEL_SIZE, padding=TEMPORAL_PADDING)
        self.ssm = SelectiveSSM(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        up = self.upper(x)

        low = x.transpose(1, 2)
        low = self.conv(low)
        low = low[:, :, :x.shape[1]]
        low = low.transpose(1, 2)

        low = F.silu(low)
        low = self.ssm(low)

        out = up * low
        out = self.proj(out)
        out = out + residual
        return out


class MotionAwareBiMambaBlock(nn.Module):


    def __init__(self, dim=FEATURE_DIM, dropout=MOTION_DROPOUT):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.delta_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.delta_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.fwd_mamba = MambaBlock(dim)
        self.bwd_mamba = MambaBlock(dim)
        self.dir_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        residual = x
        x_norm = self.pre_norm(x)

        delta = torch.zeros_like(x_norm)
        delta[:, 1:] = x_norm[:, 1:] - x_norm[:, :-1]
        delta_feat = self.delta_proj(delta)
        motion_gate = self.delta_gate(torch.cat([x_norm, delta.abs()], dim=-1))
        x_motion = x_norm + motion_gate * delta_feat

        fwd = self.fwd_mamba(x_motion)
        bwd = torch.flip(self.bwd_mamba(torch.flip(x_motion, dims=[1])), dims=[1])
        gate = self.dir_gate(torch.cat([fwd, bwd], dim=-1))
        fused = gate * fwd + (1.0 - gate) * bwd

        fused = self.out_proj(self.out_norm(fused))
        return residual + fused


class TemporalAttentionPooling(nn.Module):


    def __init__(self, dim=FEATURE_DIM):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
        self.recency_strength = nn.Parameter(torch.tensor(RECENCY_INIT))

    def forward(self, x):
        B, T, _ = x.shape
        scores = self.score(x).squeeze(-1)
        recency = torch.linspace(-1.0, 1.0, T, device=x.device, dtype=x.dtype).unsqueeze(0)
        scores = scores + self.recency_strength * recency
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1)


class LaneIntentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = ResEncoder(3)
        self.flow_encoder = ResEncoder(2)
        self.cross = CrossAttentionFusion()
        self.mamba = MotionAwareBiMambaBlock()
        self.temporal_pool = TemporalAttentionPooling()
        self.dropout = nn.Dropout(CLASSIFIER_DROPOUT)
        self.fc = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def forward(self, rgb, flow):
        B, T, C, H, W = rgb.shape
        rgb_feat = self.rgb_encoder(rgb)
        flow_feat = self.flow_encoder(flow)

        feat = self.cross(rgb_feat, flow_feat, B, T)
        feat = self.mamba(feat)

        feat = self.temporal_pool(feat)
        feat = self.dropout(feat)
        out = self.fc(feat)
        return out


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, rgb, flow, target_class=None):
        self.handlers.append(self.target_layer.register_forward_hook(self.save_activation))
        self.handlers.append(self.target_layer.register_full_backward_hook(self.save_gradient))

        self.model.eval()
        self.model.zero_grad()

        out = self.model(rgb, flow)
        if target_class is None:
            target_class = out.argmax(dim=1).item()

        score = out[0, target_class]
        score.backward()

        grads = self.gradients
        acts = self.activations
        bt, c, h, w = acts.size()

        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_activations = torch.sum(acts, dim=(2, 3), keepdim=True)

        eps = 1e-7
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
        weights = torch.sum(aij * torch.relu(grads), dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.view(bt, -1)
        cam -= cam.min(dim=1, keepdim=True)[0]
        cam /= (cam.max(dim=1, keepdim=True)[0] + eps)
        cam = cam.view(bt, h, w).cpu().detach().numpy()

        for handle in self.handlers:
            handle.remove()
        self.handlers.clear()

        return cam, target_class


def denormalize_rgb(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    return np.uint8(img * 255)


def run_epoch(loader, model, criterion, optimizer=None, device="cuda"):
    train = optimizer is not None
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            rgb, flow, label = batch[:3]
            rgb = rgb.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            pred = model(rgb, flow)
            loss = criterion(pred, label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += loss.item()
            p = pred.argmax(dim=1)
            correct += (p == label).sum().item()
            total += label.size(0)

    return total_loss / len(loader), correct / total


def get_predictions(loader, model, device="cuda", return_metadata=False):


    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_video_names = []

    with torch.no_grad():
        for batch in loader:
            rgb, flow, label = batch[:3]
            video_names = batch[3] if len(batch) > 3 else None
            rgb = rgb.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)

            out = model(rgb, flow)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if video_names is not None:
                all_video_names.extend(list(video_names))

    outputs = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    if return_metadata:
        return (*outputs, all_video_names)
    return outputs


def profile_model_hardware(model, device=DEVICE, seq_len=SEQ_LEN, img_size=IMG_SIZE):


    model.eval()

    dummy_rgb = torch.randn(1, seq_len, 3, img_size, img_size).to(device)
    dummy_flow = torch.randn(1, seq_len, 2, img_size, img_size).to(device)


    try:
        from thop import profile
        flops, params = profile(model, inputs=(dummy_rgb, dummy_flow), verbose=False)
    except ImportError:
        print("\n[警告] 未安装 thop 库，无法计算 FLOPs。请使用 pip install thop 安装。")
        flops = 0
        params = sum(p.numel() for p in model.parameters())


    with torch.no_grad():
        for _ in range(PROFILE_WARMUP_STEPS):
            _ = model(dummy_rgb, dummy_flow)

    iterations = PROFILE_ITERATIONS
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    with torch.no_grad():
        for i in range(iterations):
            start_events[i].record()
            _ = model(dummy_rgb, dummy_flow)
            end_events[i].record()


    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_latency_ms = sum(times_ms) / iterations
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0

    return params, flops, avg_latency_ms, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane-change intention experiments")
    parser.add_argument("--early-warning", action="store_true",
                        help="Run the fixed 30-frame t_c lead-time experiment.")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate RGB-flow alignment and event-level splits without training.")
    args, _ = parser.parse_known_args()
    if args.early_warning:
        from early_warning_tc_experiment import run_early_warning_experiment

        run_early_warning_experiment(
            dataset_cls=LaneDataset, model_cls=LaneIntentionModel, run_epoch_fn=run_epoch,
            prediction_fn=get_predictions, set_seed_fn=set_seed, rgb_root=RGB_ROOT, flow_root=FLOW_ROOT,
            output_dir=EARLY_WARNING_OUTPUT_DIR, seq_len=SEQ_LEN, img_size=IMG_SIZE,
            batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            num_workers=NUM_WORKERS, device=DEVICE, seeds=SEEDS[:NUM_RUNS],
            window_specs=EARLY_WARNING_WINDOWS, event_frame_count=EVENT_FRAME_COUNT,
            train_transform=transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ColorJitter(brightness=BRIGHTNESS_JITTER, contrast=CONTRAST_JITTER, saturation=SATURATION_JITTER, hue=HUE_JITTER),
                transforms.ToTensor(),
            ]),
            eval_transform=transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
            ]),
            validate_only=args.validate_only,
        )
        raise SystemExit(0)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_fscore_support
    import pandas as pd

    if len(SEEDS) < NUM_RUNS:
        raise ValueError("The SEEDS list must have at least as many seeds as NUM_RUNS.")

    print(f"Current SEQ_LEN configuration: {SEQ_LEN}")
    print("Preparing dataset list...")
    folders = sorted(glob.glob(os.path.join(RGB_ROOT, "*")))


    all_labels = []
    for folder_path in folders:
        video_name = os.path.basename(folder_path)
        if video_name.startswith("L"):
            all_labels.append(0)
        elif video_name.startswith("R"):
            all_labels.append(1)
        elif video_name.startswith("K"):
            all_labels.append(2)
        else:
            all_labels.append(0)


    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=BRIGHTNESS_JITTER, contrast=CONTRAST_JITTER, saturation=SATURATION_JITTER, hue=HUE_JITTER),
        transforms.ToTensor()
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    all_runs_results = []
    all_runs_epoch_histories = []
    all_runs_cms = []
    all_runs_rocs = []


    global_y_true_pool = []
    global_y_probs_pool = []


    hardware_stats = {}


    for run in range(NUM_RUNS):

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        current_seed = SEEDS[run]
        run_name = f"Run_{run + 1}"


        set_seed(current_seed)

        print(f"\n{'=' * 50}")
        print(f"Starting {run_name}/{NUM_RUNS} with Random Seed: {current_seed}")
        print(f"{'=' * 50}")


        train_files, temp, train_labels, temp_labels = train_test_split(
            folders, all_labels, test_size=HOLDOUT_RATIO, random_state=current_seed, stratify=all_labels
        )
        val_files, test_files, _, _ = train_test_split(
            temp, temp_labels, test_size=TEST_RATIO_IN_HOLDOUT, random_state=current_seed, stratify=temp_labels
        )

        train_ds = LaneDataset(train_files, FLOW_ROOT, SEQ_LEN, train_transform)
        val_ds = LaneDataset(val_files, FLOW_ROOT, SEQ_LEN, val_test_transform)
        test_ds = LaneDataset(test_files, FLOW_ROOT, SEQ_LEN, val_test_transform)

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


        model = LaneIntentionModel().to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


        if run == 0:
            print("\nProfiling Model Hardware Performance...")
            params, flops, latency, fps = profile_model_hardware(model, DEVICE, SEQ_LEN, IMG_SIZE)
            hardware_stats = {
                "Parameters (M)": params / 1e6,
                "FLOPs (G)": flops / 1e9,
                "Latency (ms)": latency,
                "FPS": fps
            }
            print(f"Params: {hardware_stats['Parameters (M)']:.2f} M, FLOPs: {hardware_stats['FLOPs (G)']:.2f} G")
            print(f"Latency: {latency:.2f} ms, FPS: {fps:.2f}")


        save_model_path = os.path.join(OUTPUT_DIR, f"best_model_seq{SEQ_LEN}_{MODEL_TAG}_{run_name}_seed{current_seed}.pth")

        history = []
        best_loss = float('inf')
        start_epoch = 0


        if os.path.exists(save_model_path):
            print(f"\n[发现历史检查点] 加载 {run_name} 的断点文件: {save_model_path}")
            checkpoint = torch.load(save_model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                pass
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            history = checkpoint.get('history', [])
        else:
            print(f"\n[全新训练] 开始 {run_name} 的训练。")

        print(f"\nTraining Loop for {run_name}...")
        for epoch in range(start_epoch, EPOCHS):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n[{run_name}] Epoch {epoch + 1}/{EPOCHS} [LR: {current_lr:.6f}]")

            train_loss, train_acc = run_epoch(train_loader, model, criterion, optimizer, DEVICE)
            val_loss, val_acc = run_epoch(val_loader, model, criterion, None, DEVICE)

            scheduler.step()

            history.append([epoch + 1, train_loss, train_acc, val_loss, val_acc])
            print(
                f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'history': history
                }
                torch.save(checkpoint, save_model_path)
                print(f"=> Saved new best model for {run_name}! (Val Loss dropped to {best_loss:.4f})")

        all_runs_epoch_histories.append(history)


        print(f"\nLoading best model of {run_name} for Final Test...")
        checkpoint = torch.load(save_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc = run_epoch(test_loader, model, criterion, None, DEVICE)


        y_true, y_pred, y_probs = get_predictions(test_loader, model, DEVICE)


        global_y_true_pool.extend(y_true.tolist())
        global_y_probs_pool.extend(y_probs.tolist())


        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)


        try:
            auc_score = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except ValueError:
            auc_score = float('nan')

        print(
            f"---> {run_name} Final Metrics | Acc: {test_acc:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")


        all_runs_results.append({
            "Run": run_name,
            "Seed": current_seed,
            "Test_Loss": test_loss,
            "Accuracy_Pct": test_acc * 100,
            "Precision_Pct": precision * 100,
            "Recall_Pct": recall * 100,
            "F1_Score_Pct": f1 * 100,
            "AUC_Pct": auc_score * 100
        })


        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        cm_df = pd.DataFrame(cm,
                             index=["True_Left", "True_Right", "True_Keep"],
                             columns=["Pred_Left", "Pred_Right", "Pred_Keep"])
        all_runs_cms.append((run_name, cm_df))


        class_map_name = {0: "Left", 1: "Right", 2: "Keep"}
        run_roc_data = []
        for class_idx in range(3):
            y_true_binary = (y_true == class_idx).astype(int)
            y_scores = y_probs[:, class_idx]
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
                for f, t, th in zip(fpr, tpr, thresholds):
                    run_roc_data.append({
                        "Run": run_name,
                        "Class": class_map_name[class_idx],
                        "FPR": f,
                        "TPR": t,
                        "Threshold": th
                    })
        all_runs_rocs.append(pd.DataFrame(run_roc_data))


        print(f"\nStarting Grad-CAM++ Visualization for {run_name}...")
        cam_extractor = GradCAMPlusPlus(model, model.cross.cam_target)

        random.seed(CAM_RANDOM_SEED)
        test_indices = random.sample(range(len(test_ds)), min(CAM_SAMPLE_COUNT, len(test_ds)))
        class_map = {0: "Left", 1: "Right", 2: "Keep"}

        for sample_count, dataset_idx in enumerate(test_indices):
            rgb_seq, flow_seq, sample_label = test_ds[dataset_idx]
            sample_rgb = rgb_seq.unsqueeze(0).to(DEVICE)
            sample_flow = flow_seq.unsqueeze(0).to(DEVICE)

            cam_sequence, pred_class = cam_extractor.generate(sample_rgb, sample_flow, target_class=None)

            safe_true = class_map[sample_label]
            safe_pred = class_map[pred_class]


            folder_name = f"[{run_name}] Sample {sample_count + 1} - True-{safe_true} - Pred-{safe_pred}"
            sample_dir = os.path.join(OUTPUT_DIR, "GradCAM_Frames", folder_name)
            os.makedirs(sample_dir, exist_ok=True)

            MAX_COLS = CAM_MAX_COLUMNS
            cols = min(SEQ_LEN, MAX_COLS)
            groups = int(np.ceil(SEQ_LEN / MAX_COLS))
            rows = groups * 2

            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 3, rows * 3.5))

            if rows == 2 and cols == 1:
                axes = np.array([[axes[0]], [axes[1]]])
            elif rows == 2:
                pass
            else:
                axes = np.atleast_2d(axes)

            for frame_idx in range(SEQ_LEN):
                group_idx = frame_idx // MAX_COLS
                col_idx = frame_idx % MAX_COLS
                row_orig = group_idx * 2
                row_cam = group_idx * 2 + 1

                original_img = denormalize_rgb(sample_rgb[0, frame_idx])

                heatmap = cam_sequence[frame_idx]
                heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

                superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)


                orig_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
                cam_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(sample_dir, f"orig_F{frame_idx + 1}.png"), orig_bgr)
                cv2.imwrite(os.path.join(sample_dir, f"cam_F{frame_idx + 1}.png"), cam_bgr)

                ax_orig = axes[row_orig, col_idx]
                ax_orig.imshow(original_img)
                ax_orig.set_title(f"F{frame_idx + 1}", fontsize=12)
                ax_orig.axis('off')

                ax_cam = axes[row_cam, col_idx]
                ax_cam.imshow(superimposed_img)
                ax_cam.set_title("CAM", fontsize=12)
                ax_cam.axis('off')

            for empty_idx in range(SEQ_LEN, groups * MAX_COLS):
                group_idx = empty_idx // MAX_COLS
                col_idx = empty_idx % MAX_COLS
                axes[group_idx * 2, col_idx].axis('off')
                axes[group_idx * 2 + 1, col_idx].axis('off')

            fig.suptitle(
                f"[{run_name}] Sample {sample_count + 1} | True: {class_map[sample_label]} | Pred: {class_map[pred_class]}",
                fontsize=20, y=1.02)
            plt.tight_layout()


            vis_path = os.path.join(sample_dir, "combined_overview.png")
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close(fig)


    metrics_to_calc = ["Accuracy_Pct", "Precision_Pct", "Recall_Pct", "F1_Score_Pct", "AUC_Pct"]
    final_stats_data = []

    print("\n" + "=" * 50)
    for metric in metrics_to_calc:
        vals = [res[metric] for res in all_runs_results if not np.isnan(res[metric])]
        mean_val = np.mean(vals) if len(vals) > 0 else 0
        std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0


        paper_str = f"{mean_val:.2f} ± {std_val:.2f}"
        print(f"{metric.replace('_Pct', '')}: {paper_str}")

        final_stats_data.append({
            "Metric": metric,
            "Mean": mean_val,
            "Std": std_val,
            "Paper_Format": paper_str
        })
    print("=" * 50 + "\n")


    global_y_true_np = np.array(global_y_true_pool)
    global_y_probs_np = np.array(global_y_probs_pool)

    global_roc_data = []
    class_map_name = {0: "Left", 1: "Right", 2: "Keep"}

    for class_idx in range(3):
        y_true_binary = (global_y_true_np == class_idx).astype(int)
        y_scores = global_y_probs_np[:, class_idx]

        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
            for f, t, th in zip(fpr, tpr, thresholds):

                if np.isinf(th):
                    th = 1.1
                global_roc_data.append({
                    "Class": class_map_name[class_idx],
                    "FPR": f,
                    "TPR": t,
                    "Threshold": th
                })

    global_roc_df = pd.DataFrame(global_roc_data)


    print(f"Exporting massive results to {EXCEL_NAME}...")
    with pd.ExcelWriter(EXCEL_NAME) as writer:

        summary_df = pd.DataFrame(all_runs_results)
        summary_df.to_excel(writer, sheet_name="Metrics_Summary", index=False)


        stats_df = pd.DataFrame(final_stats_data)
        stats_df.to_excel(writer, sheet_name="Final_Statistics", index=False)


        hw_df = pd.DataFrame([hardware_stats])
        hw_df.to_excel(writer, sheet_name="Hardware_Performance", index=False)


        start_row = 0
        for r_name, cm_df in all_runs_cms:
            pd.DataFrame([[f"=== {r_name} ==="]]).to_excel(writer, sheet_name="Confusion_Matrices", startrow=start_row,
                                                           header=False, index=False)
            cm_df.to_excel(writer, sheet_name="Confusion_Matrices", startrow=start_row + 1)
            start_row += cm_df.shape[0] + 4


        if len(all_runs_rocs) > 0:
            final_roc_df = pd.concat(all_runs_rocs, ignore_index=True)
            final_roc_df.to_excel(writer, sheet_name="ROC_Data_SingleRuns", index=False)


        if not global_roc_df.empty:
            global_roc_df.to_excel(writer, sheet_name="Global_Smooth_ROC", index=False)


        for r_idx, history_data in enumerate(all_runs_epoch_histories):
            epoch_df = pd.DataFrame(history_data, columns=["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"])
            epoch_df.to_excel(writer, sheet_name=f"Run_{r_idx + 1}_Epochs", index=False)

    print("Process Finished Successfully! Your data for Q2 Journal is ready.")
