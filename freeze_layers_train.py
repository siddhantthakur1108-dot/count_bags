
#  Potato Bag Detection — 3-Phase Training

from ultralytics import YOLO

# ── CONFIG for YOUR dataset ──────────────────────────────────
CONFIG = {
    "data_yaml"        : "potato_bag.yaml",
    "pretrained_model" : "yolov8s.pt",   # ← small model for tiny dataset
                                          #   yolov8m if you have RTX 3060+
    "imgsz"            : 640,            # 640 is fine for conveyor belt
                                          # use 1280 only if bags are very small
    "project_dir"      : "runs/potato_bag",
    "experiment_name"  : "v1",

    # ── Phase 1: HEAD only ─────────────────────────────────
    # Short — single class head converges fast
    "phase1": {
        "epochs"        : 30,
        "freeze"        : 10,      # full backbone frozen
        "lr0"           : 0.005,   # lower than usual — small dataset
        "lrf"           : 0.0005,
        "batch"         : 16,
        "optimizer"     : "AdamW",
        "warmup_epochs" : 3,
        "patience"      : 8,       # stop early if no improvement for 8 epochs
    },

    # ── Phase 2: NECK + HEAD ───────────────────────────────
    # Let neck adapt to conveyor belt object scales
    "phase2": {
        "epochs"        : 20,
        "freeze"        : 3,       # only freeze first 3 backbone layers
        "lr0"           : 0.0005,
        "lrf"           : 0.00005,
        "batch"         : 16,
        "optimizer"     : "AdamW",
        "warmup_epochs" : 2,
        "patience"      : 7,
    },

    # ── Phase 3: Full unfreeze ─────────────────────────────
    # Short + very low LR — overfit risk is high with 550 images
    "phase3": {
        "epochs"        : 20,
        "freeze"        : 0,
        "lr0"           : 0.00005,  # very conservative
        "lrf"           : 0.000005,
        "batch"         : 16,
        "optimizer"     : "AdamW",
        "warmup_epochs" : 1,
        "patience"      : 6,        # stop quickly if overfitting starts
    },

    "device"     : "0",
    "workers"    : 4,
    "amp"        : True,
    "multi_scale": False,  # OFF for small dataset — adds training noise
}


def train_potato_bag():

    # ── PHASE 1 ────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1 — Training HEAD only (30 epochs)")
    print("Backbone + Neck: FROZEN | LR = 0.005")
    print("="*60)

    model = YOLO(CONFIG["pretrained_model"])

    p1 = CONFIG["phase1"]
    model.train(
        data          = CONFIG["data_yaml"],
        epochs        = p1["epochs"],          # 30 epochs
        imgsz         = CONFIG["imgsz"],
        batch         = p1["batch"],
        freeze        = p1["freeze"],          # freeze layers 0–9
        lr0           = p1["lr0"],
        lrf           = p1["lrf"],
        optimizer     = p1["optimizer"],
        warmup_epochs = p1["warmup_epochs"],
        cos_lr        = True,
        patience      = p1["patience"],        # early stop at 8 epochs no improve
        weight_decay  = 0.0005,

        # ── Augmentation — aggressive for small dataset ──
        mosaic        = 1.0,
        copy_paste    = 0.3,   # paste bags onto empty belt images ← key
        mixup         = 0.15,
        degrees       = 10.0,
        translate     = 0.15,
        scale         = 0.8,
        fliplr        = 0.5,
        flipud        = 0.0,
        hsv_h         = 0.02,
        hsv_s         = 0.8,
        hsv_v         = 0.5,
        perspective   = 0.001,

        device        = CONFIG["device"],
        workers       = CONFIG["workers"],
        amp           = CONFIG["amp"],
        multi_scale   = CONFIG["multi_scale"],
        plots         = True,
        save_period   = 5,
        project       = CONFIG["project_dir"],
        name          = "phase1",
        exist_ok      = True,
    )
    print("Phase 1 complete. Best weights → runs/potato_bag/phase1/weights/best.pt")


    # ── PHASE 2 ────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2 — Training NECK + HEAD (20 epochs)")
    print("Layers 0-2: FROZEN | LR = 0.0005")
    print("="*60)

    model = YOLO("runs/potato_bag/phase1/weights/best.pt")

    p2 = CONFIG["phase2"]
    model.train(
        data          = CONFIG["data_yaml"],
        epochs        = p2["epochs"],          # 20 epochs
        imgsz         = CONFIG["imgsz"],
        batch         = p2["batch"],
        freeze        = p2["freeze"],          # only freeze layers 0–2
        lr0           = p2["lr0"],
        lrf           = p2["lrf"],
        optimizer     = p2["optimizer"],
        warmup_epochs = p2["warmup_epochs"],
        cos_lr        = True,
        patience      = p2["patience"],
        weight_decay  = 0.0005,

        mosaic        = 1.0,
        copy_paste    = 0.3,
        mixup         = 0.1,
        degrees       = 10.0,
        translate     = 0.15,
        scale         = 0.8,
        fliplr        = 0.5,
        flipud        = 0.0,
        hsv_h         = 0.02,
        hsv_s         = 0.8,
        hsv_v         = 0.5,

        device        = CONFIG["device"],
        workers       = CONFIG["workers"],
        amp           = CONFIG["amp"],
        multi_scale   = CONFIG["multi_scale"],
        plots         = True,
        save_period   = 5,
        project       = CONFIG["project_dir"],
        name          = "phase2",
        exist_ok      = True,
    )
    print("Phase 2 complete. Best weights → runs/potato_bag/phase2/weights/best.pt")


    # ── PHASE 3 ────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 3 — Full network (20 epochs)")
    print("Nothing frozen | LR = 0.00005 (very gentle)")
    print("="*60)

    model = YOLO("runs/potato_bag/phase2/weights/best.pt")

    p3 = CONFIG["phase3"]
    model.train(
        data          = CONFIG["data_yaml"],
        epochs        = p3["epochs"],          # 20 epochs
        imgsz         = CONFIG["imgsz"],
        batch         = p3["batch"],
        freeze        = p3["freeze"],          # 0 — nothing frozen
        lr0           = p3["lr0"],
        lrf           = p3["lrf"],
        optimizer     = p3["optimizer"],
        warmup_epochs = p3["warmup_epochs"],
        cos_lr        = True,
        patience      = p3["patience"],
        weight_decay  = 0.001,                # higher weight decay in p3

        # Reduce augmentation in phase 3 — let model settle
        mosaic        = 0.5,
        copy_paste    = 0.1,
        mixup         = 0.05,
        degrees       = 5.0,
        translate     = 0.1,
        scale         = 0.5,
        fliplr        = 0.5,
        hsv_h         = 0.02,
        hsv_s         = 0.5,
        hsv_v         = 0.3,

        device        = CONFIG["device"],
        workers       = CONFIG["workers"],
        amp           = CONFIG["amp"],
        multi_scale   = CONFIG["multi_scale"],
        plots         = True,
        save_period   = 5,
        project       = CONFIG["project_dir"],
        name          = "phase3",
        exist_ok      = True,
    )
    print("Phase 3 complete.")


    # ── EVALUATE ───────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATING final model on test set")
    print("="*60)

    final_model = YOLO("runs/potato_bag/phase3/weights/best.pt")
    metrics = final_model.val(
        data     = CONFIG["data_yaml"],
        split    = "test",
        imgsz    = CONFIG["imgsz"],
        device   = CONFIG["device"],
        plots    = True,
        save_json= True,
        project  = CONFIG["project_dir"],
        name     = "final_eval",
    )

    print(f"\n  mAP@50     : {metrics.box.map50:.4f}")
    print(f"  mAP@50:95  : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.p.mean():.4f}")
    print(f"  Recall     : {metrics.box.r.mean():.4f}")

    # Export for deployment
    final_model.export(format="onnx", imgsz=CONFIG["imgsz"])
    print("\nTraining pipeline complete!")
    print("Final weights: runs/potato_bag/phase3/weights/best.pt")


if __name__ == "__main__":
    train_potato_bag()