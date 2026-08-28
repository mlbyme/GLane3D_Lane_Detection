from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)

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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Smoke-test settings.
EPOCHS = 1
MAX_SAMPLES = 100
LOG_INTERVAL = 10

LEARNING_RATE = 3e-4
ACCUMULATION_STEPS = 8

USE_AMP = True
AMP_DTYPE = torch.bfloat16


transform = Compose([
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def collate_one(batch):
    return batch[0]


def prepare_sample(sample):
    image = transform(
        sample["image"]
    ).unsqueeze(0)

    image = image.to(
        DEVICE,
        non_blocking=True,
    )

    intrinsic = torch.tensor(
        sample["intrinsic"],
        dtype=torch.float32,
        device=DEVICE,
    )

    extrinsic = torch.tensor(
        sample["extrinsic"],
        dtype=torch.float32,
        device=DEVICE,
    )

    return image, intrinsic, extrinsic


def compute_losses(
    model,
    criterion,
    sample,
    image,
    intrinsic,
    extrinsic,
):
    row_forward = model.bev_grid[:, 0, 0]

    gt = build_gt_keypoints(
        sample["lane_lines"],
        row_forward,
        extrinsic,
    )

    if gt is None:
        return None

    proposal_target = build_proposal_target(
        sample["lane_lines"],
        model.bev_grid,
        extrinsic,
        positive_radius=2.0,
    ).unsqueeze(0)

    output = model(
        image,
        intrinsic,
        extrinsic,
    )

    proposal_rows = (
        output["proposal_indices"][0]
        // model.bev_grid.shape[1]
    )

    # First Hungarian pass:
    # all 512 proposals, GT repeated twice.
    matched_pred_1, matched_gt_1 = match_keypoints(
        anchors=output["proposals"][0],
        refined_points=output["refined_points"][0],
        gt=gt,
        proposal_rows=proposal_rows,
        class_logits=output["class_logits"][0],
        repeat=2,
    )

    keep = output["keep_indices"]

    # Second Hungarian pass:
    # strongest points after PointNMS, no GT repetition.
    matched_pred_2, matched_gt_2 = match_keypoints(
        anchors=output["proposals"][0, keep],
        refined_points=output["strong_points"][0],
        gt=gt,
        proposal_rows=proposal_rows[keep],
        class_logits=output["class_logits"][0, keep],
        repeat=1,
    )

    connection_target = build_connection_targets(
        matched_pred_2,
        matched_gt_2,
        gt,
        num_predictions=len(
            output["strong_points"][0]
        ),
    )

    losses = criterion(
        seg_logits=output["seg_logits"],
        seg_target=proposal_target,
        x_offset=output["x_offset"][0],
        z=output["z"][0],
        class_logits=output["class_logits"][0],
        matched_pred=matched_pred_1,
        matched_gt=matched_gt_1,
        proposal_anchors=output["proposals"][0],
        gt_points=gt["points"],
        gt_classes=gt["categories"],
        adjacency_logits=output["adjacency_logits"][0],
        adjacency_target=connection_target,
    )

    losses["matches_1"] = len(
        matched_pred_1
    )

    losses["matches_2"] = len(
        matched_pred_2
    )

    losses["connections"] = int(
        connection_target.sum().item()
    )

    losses["gt_keypoints"] = len(
        gt["points"]
    )

    return losses


def optimizer_step(
    model,
    criterion,
    optimizer,
    scaler,
    gradient_scale=1.0,
):
    scaler.unscale_(optimizer)

    # Needed only when the final accumulation group
    # contains fewer than ACCUMULATION_STEPS samples.
    if gradient_scale != 1.0:
        for parameter in list(
            model.parameters()
        ) + list(
            criterion.parameters()
        ):
            if parameter.grad is not None:
                parameter.grad.mul_(
                    gradient_scale
                )

    torch.nn.utils.clip_grad_norm_(
        list(model.parameters())
        + list(criterion.parameters()),
        max_norm=5.0,
    )

    scaler.step(optimizer)
    scaler.update()

    #TEMP

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(
            f"Non-finite parameter after optimizer step: {name}"
        )


    optimizer.zero_grad(
        set_to_none=True
    )


def main():
    dataset = OpenLaneDataset(
        DATA_ROOT,
        split="training",
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=collate_one,
    )

    model = GLane3D().to(DEVICE)
    criterion = GLane3DLoss().to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(criterion.parameters()),
        lr=LEARNING_RATE,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,
    )

    print("Device:", DEVICE)
    print("Training samples:", len(dataset))
    print("Smoke-test samples:", MAX_SAMPLES)
    print(
        "Gradient accumulation:",
        ACCUMULATION_STEPS,
    )
    print()

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0
        trained_samples = 0
        samples_since_step = 0

        for sample in loader:
            image, intrinsic, extrinsic = (
                prepare_sample(sample)
            )

            with torch.autocast(
                device_type="cuda",
                dtype=AMP_DTYPE,
                enabled=USE_AMP,
            ):
                losses = compute_losses(
                    model,
                    criterion,
                    sample,
                    image,
                    intrinsic,
                    extrinsic,
                )

                if losses is None:
                    continue

                if not torch.isfinite(losses["total"]):
                     print("\nNon-finite loss detected")

                     for name in [
                         "total",
                         "keypoint",
                         "regression",
                         "connection",
                         "classification",
                     ]:
                         value = losses[name]
                         print(
                             name,
                             value.detach().float().item(),
                         )
                 
                     raise RuntimeError(
                         "Training stopped because loss became non-finite"
                     )

                loss = (
                    losses["total"]
                    / ACCUMULATION_STEPS
                )

            scaler.scale(
                loss
            ).backward()

            trained_samples += 1
            samples_since_step += 1

            running_loss += (
                losses["total"].item()
            )

            if (
                samples_since_step
                == ACCUMULATION_STEPS
            ):
                optimizer_step(
                    model,
                    criterion,
                    optimizer,
                    scaler,
                )

                samples_since_step = 0

            if (
                trained_samples
                % LOG_INTERVAL
                == 0
            ):
                average_loss = (
                    running_loss
                    / trained_samples
                )

                print(
                    f"epoch {epoch + 1:02d} "
                    f"sample {trained_samples:04d} "
                    f"loss {average_loss:.4f} "
                    f"kp {losses['keypoint'].item():.3f} "
                    f"reg {losses['regression'].item():.3f} "
                    f"conn {losses['connection'].item():.3f} "
                    f"cls {losses['classification'].item():.3f} "
                    f"gt {losses['gt_keypoints']} "
                    f"m1 {losses['matches_1']} "
                    f"m2 {losses['matches_2']} "
                    f"edges {losses['connections']}"
                )

            if trained_samples >= MAX_SAMPLES:
                break

        # Apply gradients from the final partial
        # accumulation group.
        if samples_since_step > 0:
            gradient_scale = (
                ACCUMULATION_STEPS
                / samples_since_step
            )

            optimizer_step(
                model,
                criterion,
                optimizer,
                scaler,
                gradient_scale=gradient_scale,
            )

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        checkpoint_path = (
            CHECKPOINT_DIR
            / "glane3d_smoke.pt"
        )

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        print()
        print(
            "Saved checkpoint:",
            checkpoint_path,
        )

    elapsed = time.time() - start_time

    print()
    print("Smoke training complete")
    print("Samples:", trained_samples)

    print(
        "Elapsed time:",
        f"{elapsed:.2f} seconds",
    )

    print(
        "Seconds per sample:",
        f"{elapsed / max(trained_samples, 1):.3f}",
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