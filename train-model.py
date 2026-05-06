from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')

    results = model.train(
        # ── Core Setup ──────────────────────────────────────────────
        data        = 'C:/Users/siddh/Desktop/count_begs_ver1/count_begs.v4i.yolov8/data.yaml',
        epochs      = 200,
        patience    = 50,
        batch       = 8,            # ✅ Keep 8 for RTX 3050 6GB
        imgsz       = 640,
        device      = 0,

        # ── Optimizer ───────────────────────────────────────────────
        optimizer        = 'AdamW',
        lr0              = 0.001,
        lrf              = 0.01,
        momentum         = 0.937,
        weight_decay     = 0.0005,
        warmup_epochs    = 5,
        warmup_momentum  = 0.8,
        warmup_bias_lr   = 0.1,

        # ── Loss Weights ─────────────────────────────────────────────
        box         = 7.5,
        cls         = 0.5,
        dfl         = 1.5,

        # ── Regularization ───────────────────────────────────────────
        dropout     = 0.1,

        # ── Workers: KEY FIX for Windows ─────────────────────────────
        workers     = 0,            # ✅ Set to 0 on Windows to avoid spawn issues

        # ── Saving & Logging ─────────────────────────────────────────
        val         = True,
        save        = True,
        save_period = 10,
        plots       = True,
        exist_ok    = True,
        project     = 'potato_bag_detection',
        name        = 'yolov8m_run1',
        seed        = 42,
        deterministic = True,
    )