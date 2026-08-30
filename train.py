from pathlib import Path
import math
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from datasets.openlane import OpenLaneDataset
from datasets.targets import (
    build_connection_targets,
    build_gt_keypoints,
    build_proposal_target,
)
from models.glane3d import GLane3D
from models.losses import GLane3DLoss
from models.matcher import match_keypoints


DATA_ROOT = (
    "/home/hp/datasets/openlane/"
    "openlane_v1_300"
)

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Full Lane300 training settings.
EPOCHS = 24
MAX_SAMPLES = None
LOG_INTERVAL = 200

LEARNING_RATE = 3e-4

BATCH_SIZE = 4
ACCUMULATION_STEPS = 4

USE_AMP = True
AMP_DTYPE = torch.bfloat16

# Paper specifies warm-up + cosine annealing,
# but does not specify the exact warm-up duration.
WARMUP_EPOCHS = 1
WARMUP_START_FACTOR = 0.1

RUN_NAME = "glane3d_lane300_full24"
NUM_WORKERS = 4


transform = Compose([
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def collate_batch(batch):
    images = torch.stack([
        transform(sample["image"])
        for sample in batch
    ])

    intrinsics = torch.tensor(
        [sample["intrinsic"] for sample in batch],
        dtype=torch.float32,
    )

    extrinsics = torch.tensor(
        [sample["extrinsic"] for sample in batch],
        dtype=torch.float32,
    )

    return {
        "image": images,
        "intrinsic": intrinsics,
        "extrinsic": extrinsics,
        "lane_lines": [
            sample["lane_lines"]
            for sample in batch
        ],
        "image_path": [
            sample["image_path"]
            for sample in batch
        ],
    }


def prepare_batch(batch):
    image = batch["image"].to(
        DEVICE,
        non_blocking=True,
    )

    intrinsic = batch["intrinsic"].to(
        DEVICE,
        non_blocking=True,
    )

    extrinsic = batch["extrinsic"].to(
        DEVICE,
        non_blocking=True,
    )

    return image, intrinsic, extrinsic


def compute_losses(
    model,
    criterion,
    batch,
    image,
    intrinsic,
    extrinsic,
):
    output = model(
        image,
        intrinsic,
        extrinsic,
    )

    row_forward = model.bev_grid[:, 0, 0]

    batch_losses = []

    total_gt = 0
    total_m1 = 0
    total_m2 = 0
    total_edges = 0

    for b in range(image.shape[0]):
        gt = build_gt_keypoints(
            batch["lane_lines"][b],
            row_forward,
            extrinsic[b],
        )

        if gt is None:
            continue

        proposal_target = build_proposal_target(
            batch["lane_lines"][b],
            model.bev_grid,
            extrinsic[b],
            positive_radius=2.0,
        ).unsqueeze(0)

        proposal_rows = (
            output["proposal_indices"][b]
            // model.bev_grid.shape[1]
        )

        matched_pred_1, matched_gt_1 = match_keypoints(
            anchors=output["proposals"][b],
            refined_points=output["refined_points"][b],
            gt=gt,
            proposal_rows=proposal_rows,
            class_logits=output["class_logits"][b],
            repeat=2,
        )

        keep = output["keep_indices"][b]

        matched_pred_2, matched_gt_2 = match_keypoints(
            anchors=output["proposals"][b, keep],
            refined_points=output["strong_points"][b],
            gt=gt,
            proposal_rows=proposal_rows[keep],
            class_logits=output["class_logits"][b, keep],
            repeat=1,
        )

        connection_target = build_connection_targets(
            matched_pred_2,
            matched_gt_2,
            gt,
            num_predictions=len(
                output["strong_points"][b]
            ),
        )

        losses = criterion(
            seg_logits=output["seg_logits"][b:b + 1],
            seg_target=proposal_target,
            x_offset=output["x_offset"][b],
            z=output["z"][b],
            class_logits=output["class_logits"][b],
            matched_pred=matched_pred_1,
            matched_gt=matched_gt_1,
            proposal_anchors=output["proposals"][b],
            gt_points=gt["points"],
            gt_classes=gt["categories"],
            adjacency_logits=output["adjacency_logits"][b],
            adjacency_target=connection_target,
        )

        batch_losses.append(losses)

        total_gt += len(gt["points"])
        total_m1 += len(matched_pred_1)
        total_m2 += len(matched_pred_2)
        total_edges += int(
            connection_target.sum().item()
        )

    if not batch_losses:
        return None

    result = {}

    for name in [
        "total",
        "keypoint",
        "regression",
        "connection",
        "classification",
    ]:
        result[name] = torch.stack([
            item[name]
            for item in batch_losses
        ]).mean()

    result["gt"] = total_gt
    result["matches_1"] = total_m1
    result["matches_2"] = total_m2
    result["connections"] = total_edges
    result["valid_images"] = len(batch_losses)

    return result


def optimizer_step(
    model,
    criterion,
    optimizer,
    scaler,
    scheduler,
    gradient_scale=1.0,
):
    scaler.unscale_(optimizer)

    if gradient_scale != 1.0:
        parameters = (
            list(model.parameters())
            + list(criterion.parameters())
        )

        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)

    torch.nn.utils.clip_grad_norm_(
        list(model.parameters())
        + list(criterion.parameters()),
        max_norm=5.0,
    )

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(
                "Non-finite parameter after "
                f"optimizer step: {name}"
            )

    optimizer.zero_grad(set_to_none=True)


def save_checkpoint(
    model,
    criterion,
    optimizer,
    scheduler,
    epoch,
    epoch_samples,
    total_samples_seen,
    optimizer_steps,
):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch_samples": epoch_samples,
        "total_samples_seen": total_samples_seen,
        "optimizer_steps": optimizer_steps,
    }

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{RUN_NAME}_epoch_{epoch:02d}.pt"
    )

    temp_path = (
        CHECKPOINT_DIR
        / f"{RUN_NAME}_epoch_{epoch:02d}.tmp"
    )

    torch.save(checkpoint, temp_path)
    temp_path.replace(checkpoint_path)

    size_mb = (
        checkpoint_path.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"Saved checkpoint: {checkpoint_path}"
    )
    print(
        f"Checkpoint size: {size_mb:.2f} MB"
    )

    if size_mb < 10:
        raise RuntimeError(
            "Checkpoint file is unexpectedly small: "
            f"{size_mb:.2f} MB"
        )


def build_scheduler(
    optimizer,
    loader,
):
    if MAX_SAMPLES is None:
        batches_per_epoch = len(loader)
    else:
        samples_per_epoch = min(
            MAX_SAMPLES,
            len(loader.dataset),
        )
        batches_per_epoch = math.ceil(
            samples_per_epoch / BATCH_SIZE
        )

    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch
        / ACCUMULATION_STEPS
    )

    warmup_steps = (
        WARMUP_EPOCHS
        * optimizer_steps_per_epoch
    )

    total_steps = (
        EPOCHS
        * optimizer_steps_per_epoch
    )

    cosine_steps = max(
        1,
        total_steps - warmup_steps,
    )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=WARMUP_START_FACTOR,
        end_factor=1.0,
        total_iters=max(1, warmup_steps),
    )

    cosine = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    )

    return (
        scheduler,
        optimizer_steps_per_epoch,
        total_steps,
    )


def main():
    dataset = OpenLaneDataset(
        DATA_ROOT,
        split="training",
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        collate_fn=collate_batch,
    )

    model = GLane3D().to(DEVICE)
    criterion = GLane3DLoss().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(criterion.parameters()),
        lr=LEARNING_RATE,
    )

    (
        scheduler,
        optimizer_steps_per_epoch,
        total_optimizer_steps,
    ) = build_scheduler(
        optimizer,
        loader,
    )

    # BF16 does not require gradient scaling.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,
    )

    print("Device:", DEVICE)
    print("Training samples:", len(dataset))
    print("Epochs:", EPOCHS)
    print("Physical batch size:", BATCH_SIZE)
    print(
        "Gradient accumulation:",
        ACCUMULATION_STEPS,
    )
    print(
        "Effective batch size:",
        BATCH_SIZE * ACCUMULATION_STEPS,
    )
    print(
        "Optimizer steps / epoch:",
        optimizer_steps_per_epoch,
    )
    print(
        "Total optimizer steps:",
        total_optimizer_steps,
    )
    print("Warm-up epochs:", WARMUP_EPOCHS)
    print(
        "Initial LR:",
        optimizer.param_groups[0]["lr"],
    )
    print()

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    total_samples_seen = 0
    global_optimizer_step = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        trained_samples = 0
        batches_since_step = 0

        epoch_start_time = time.time()

        for batch in loader:
            image, intrinsic, extrinsic = (
                prepare_batch(batch)
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=AMP_DTYPE,
                enabled=(
                    USE_AMP
                    and DEVICE.type == "cuda"
                ),
            ):
                losses = compute_losses(
                    model,
                    criterion,
                    batch,
                    image,
                    intrinsic,
                    extrinsic,
                )

                if losses is None:
                    continue

                if not torch.isfinite(
                    losses["total"]
                ):
                    print(
                        "\nNon-finite loss detected"
                    )

                    for name in [
                        "total",
                        "keypoint",
                        "regression",
                        "connection",
                        "classification",
                    ]:
                        print(
                            name,
                            losses[name]
                            .detach()
                            .float()
                            .item(),
                        )

                    raise RuntimeError(
                        "Training stopped because "
                        "loss became non-finite"
                    )

                loss = (
                    losses["total"]
                    / ACCUMULATION_STEPS
                )

            scaler.scale(loss).backward()

            actual_batch_size = image.shape[0]

            trained_samples += actual_batch_size
            total_samples_seen += actual_batch_size
            batches_since_step += 1

            running_loss += (
                losses["total"].item()
                * actual_batch_size
            )

            if (
                batches_since_step
                == ACCUMULATION_STEPS
            ):
                optimizer_step(
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    scheduler,
                    gradient_scale=1.0,
                )

                global_optimizer_step += 1
                batches_since_step = 0

            if (
                trained_samples
                % LOG_INTERVAL
                == 0
            ):
                average_loss = (
                    running_loss
                    / max(trained_samples, 1)
                )

                current_lr = (
                    optimizer.param_groups[0]["lr"]
                )

                print(
                    f"epoch {epoch + 1:02d} "
                    f"sample {trained_samples:05d} "
                    f"loss {average_loss:.4f} "
                    f"kp {losses['keypoint'].item():.3f} "
                    f"reg {losses['regression'].item():.3f} "
                    f"conn {losses['connection'].item():.3f} "
                    f"cls {losses['classification'].item():.3f} "
                    f"lr {current_lr:.2e} "
                    f"gt {losses['gt']} "
                    f"m1 {losses['matches_1']} "
                    f"m2 {losses['matches_2']} "
                    f"edges {losses['connections']}"
                )

            if (
                MAX_SAMPLES is not None
                and trained_samples
                >= MAX_SAMPLES
            ):
                break

        if batches_since_step > 0:
            gradient_scale = (
                ACCUMULATION_STEPS
                / batches_since_step
            )

            optimizer_step(
                model,
                criterion,
                optimizer,
                scaler,
                scheduler,
                gradient_scale=gradient_scale,
            )

            global_optimizer_step += 1

        save_checkpoint(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            epoch_samples=trained_samples,
            total_samples_seen=total_samples_seen,
            optimizer_steps=global_optimizer_step,
        )

        epoch_elapsed = (
            time.time() - epoch_start_time
        )

        epoch_average_loss = (
            running_loss
            / max(trained_samples, 1)
        )

        print(
            f"Epoch {epoch + 1:02d} complete"
        )
        print("Samples:", trained_samples)
        print(
            "Average loss:",
            f"{epoch_average_loss:.4f}",
        )
        print(
            "Epoch time:",
            f"{epoch_elapsed:.2f} seconds",
        )
        print(
            "Seconds per sample:",
            f"{epoch_elapsed / max(trained_samples, 1):.3f}",
        )
        print()

    elapsed = time.time() - start_time

    print("Training complete")
    print(
        "Total samples seen:",
        total_samples_seen,
    )
    print(
        "Total optimizer steps:",
        global_optimizer_step,
    )
    print(
        "Total elapsed time:",
        f"{elapsed:.2f} seconds",
    )

    if DEVICE.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            "Peak GPU memory:",
            f"{peak_memory:.2f} GB",
        )


if __name__ == "__main__":
    main()
