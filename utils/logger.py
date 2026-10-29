import json
import csv
from datetime import datetime
from pathlib import Path

def save_history(history, save_path):
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)

def save_inference_results(output_dir, cfg, params, gflops, dice, hd95, eff):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tc_d, wt_d, et_d = dice[0].item() * 100, dice[1].item() * 100, dice[2].item() * 100
    mean_d = (tc_d + wt_d + et_d) / 3

    tc_h, wt_h, et_h = hd95[0].item(), hd95[1].item(), hd95[2].item()
    mean_h = (tc_h + wt_h + et_h) / 3

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": cfg.model,
        "params_m": round(params, 4),
        "gflops_g": round(gflops, 4),
        "dice": {
            "TC": round(tc_d, 4),
            "WT": round(wt_d, 4),
            "ET": round(et_d, 4),
            "Mean": round(mean_d, 4),
        },
        "hd95": {
            "TC": round(tc_h, 4),
            "WT": round(wt_h, 4),
            "ET": round(et_h, 4),
            "Mean": round(mean_h, 4),
        },
        "efficiency": {
            "num_cases": int(eff["num_cases"]),
            "total_time": round(float(eff["total_time"]), 4),
            "avg_time": round(float(eff["avg_time"]), 4),
            "std_time": round(float(eff["std_time"]), 4),
            "min_time": round(float(eff["min_time"]), 4),
            "max_time": round(float(eff["max_time"]), 4),
            "fps": round(float(eff["fps"]), 4),
        }
    }


    json_path = output_dir / "inference_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    csv_path = output_dir / "inference_results.csv"
    csv_header = [
        "timestamp", "model", "params_m", "gflops_g",
        "TC_Dice", "WT_Dice", "ET_Dice", "Mean_Dice",
        "TC_HD95", "WT_HD95", "ET_HD95", "Mean_HD95",
        "num_cases", "total_time", "avg_time", "std_time",
        "min_time", "max_time", "fps"
    ]
    csv_row = [
        results["timestamp"], results["model"], results["params_m"], results["gflops_g"],
        results["dice"]["TC"], results["dice"]["WT"], results["dice"]["ET"], results["dice"]["Mean"],
        results["hd95"]["TC"], results["hd95"]["WT"], results["hd95"]["ET"], results["hd95"]["Mean"],
        results["efficiency"]["num_cases"], results["efficiency"]["total_time"],
        results["efficiency"]["avg_time"], results["efficiency"]["std_time"],
        results["efficiency"]["min_time"], results["efficiency"]["max_time"],
        results["efficiency"]["fps"]
    ]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(csv_header)
        writer.writerow(csv_row)

    txt_path = output_dir / "inference_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {cfg.model}  |  Params: {params:.2f} M  |  GFLOPs: {gflops:.2f} G\n")
        f.write("=" * 70 + "\n")
        f.write("Region   Dice (%)     HD95 (voxel)\n")
        f.write("-" * 36 + "\n")
        f.write(f"ET       {et_d:.2f}        {et_h:.4f}\n")
        f.write(f"WT       {wt_d:.2f}        {wt_h:.4f}\n")
        f.write(f"TC       {tc_d:.2f}        {tc_h:.4f}\n")
        f.write("-" * 36 + "\n")
        f.write(f"Mean     {mean_d:.2f}        {mean_h:.4f}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Inference Efficiency\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Total cases:       {eff['num_cases']}\n")
        f.write(f"  Total time:        {eff['total_time']:.2f} s\n")
        f.write(f"  Avg time/case:     {eff['avg_time']:.4f} s  +/- {eff['std_time']:.4f} s\n")
        f.write(f"  Min / Max:         {eff['min_time']:.4f} s / {eff['max_time']:.4f} s\n")
        f.write(f"  FPS (volumes/s):   {eff['fps']:.4f}\n")
        f.write("=" * 70 + "\n")

    print(f"\nSaved results to:\n  {json_path}\n  {csv_path}\n  {txt_path}")
