"""
split_dataset.py - Tự động chia bộ dữ liệu Brain MRI (Kaggle LGG, kaggle_3m)
thành 3 tập độc lập: Train / Val / Test (mặc định 70/15/15).

Người phụ trách: Nguyễn Kiêm Chính

Cấu trúc đầu vào (thư mục gốc, đã giải nén từ Kaggle):
    data/kaggle_3m/
        TCGA_CS_4941_19960909/
            TCGA_CS_4941_19960909_1.tif
            TCGA_CS_4941_19960909_1_mask.tif
            ...
        TCGA_CS_4942_19970222/
            ...

Cấu trúc đầu ra (khớp với src/train.py và data/dataset.py):
    data/train/images/*.tif    data/train/masks/*_mask.tif
    data/val/images/*.tif      data/val/masks/*_mask.tif
    data/test/images/*.tif     data/test/masks/*_mask.tif
    data/splits.csv            (danh sách ảnh -> tập, dùng để tái lập / kiểm tra)

Vì sao mặc định chia THEO BỆNH NHÂN (--by patient)?
    Các lát cắt của cùng một bệnh nhân gần như giống nhau. Nếu chia ngẫu nhiên
    theo từng ảnh, lát cắt liền kề sẽ rơi vào cả Train lẫn Test (data leakage)
    -> Dice/IoU trên Test cao ảo. Chia theo bệnh nhân cho kết quả đánh giá đáng tin.
    Nếu thầy/cô yêu cầu chia theo ảnh, dùng: --by image

Cách chạy (từ thư mục gốc của project):
    python data/split_dataset.py
    python data/split_dataset.py --src data/kaggle_3m --ratio 0.7 0.15 0.15 --seed 42
    python data/split_dataset.py --by image
    python data/split_dataset.py --zip          # nén thành data/dataset_split.zip
"""

import argparse
import csv
import random
import shutil
import sys
import zipfile
from pathlib import Path

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SPLITS = ("train", "val", "test")
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Chia dataset thành Train/Val/Test.")
    p.add_argument("--src", default="data/kaggle_3m",
                   help="Thư mục gốc chứa ảnh (mặc định: data/kaggle_3m)")
    p.add_argument("--dst", default="data",
                   help="Thư mục đích tạo train/ val/ test/ (mặc định: data)")
    p.add_argument("--ratio", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Tỷ lệ chia, mặc định 0.70 0.15 0.15")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed ngẫu nhiên để cả nhóm có cùng một kết quả chia")
    p.add_argument("--by", choices=["patient", "image"], default="patient",
                   help="Chia theo bệnh nhân (khuyên dùng) hoặc theo từng ảnh")
    p.add_argument("--move", action="store_true",
                   help="Di chuyển file thay vì sao chép (tiết kiệm dung lượng)")
    p.add_argument("--zip", action="store_true",
                   help="Nén 3 tập thành dataset_split.zip để chia sẻ")
    p.add_argument("--force", action="store_true",
                   help="Xóa train/val/test cũ (nếu có) rồi chia lại")
    return p.parse_args()


def resolve(path_str):
    """Đường dẫn tương đối được hiểu là tính từ thư mục gốc của project."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def is_mask(path):
    return path.stem.endswith("_mask")


def find_pairs(src):
    """Quét toàn bộ src, trả về list (patient_id, image_path, mask_path)."""
    if not src.exists():
        sys.exit(f"[LỖI] Không tìm thấy thư mục nguồn: {src}\n"
                 f"      Hãy tải và giải nén Kaggle 'lgg-mri-segmentation' vào đó.")

    pairs, missing_mask = [], []
    for img in sorted(src.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS or is_mask(img):
            continue
        mask = img.with_name(f"{img.stem}_mask{img.suffix}")
        if not mask.exists():
            missing_mask.append(img)
            continue
        # Bệnh nhân = tên thư mục chứa ảnh (vd: TCGA_CS_4941_19960909)
        patient = img.parent.name if img.parent != src else img.stem.rsplit("_", 1)[0]
        pairs.append((patient, img, mask))

    if missing_mask:
        print(f"[CẢNH BÁO] {len(missing_mask)} ảnh không có mask, đã bỏ qua "
              f"(vd: {missing_mask[0].name})")
    if not pairs:
        sys.exit("[LỖI] Không tìm thấy cặp ảnh/mask nào.")
    return pairs


def split_counts(n, ratio):
    """Tính số phần tử mỗi tập (val/test làm tròn, train nhận phần còn lại)."""
    total = sum(ratio)
    r_train, r_val, r_test = (x / total for x in ratio)
    n_val = round(n * r_val)
    n_test = round(n * r_test)
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def assign_splits(pairs, ratio, seed, by):
    """Trả về dict: index của pair -> tên tập."""
    rng = random.Random(seed)
    assignment = {}

    if by == "patient":
        patients = sorted({p for p, _, _ in pairs})
        rng.shuffle(patients)
        n_tr, n_va, _ = split_counts(len(patients), ratio)
        patient_split = {}
        for i, pid in enumerate(patients):
            patient_split[pid] = ("train" if i < n_tr
                                  else "val" if i < n_tr + n_va else "test")
        for idx, (pid, _, _) in enumerate(pairs):
            assignment[idx] = patient_split[pid]
    else:
        idxs = list(range(len(pairs)))
        rng.shuffle(idxs)
        n_tr, n_va, _ = split_counts(len(idxs), ratio)
        for i, idx in enumerate(idxs):
            assignment[idx] = ("train" if i < n_tr
                               else "val" if i < n_tr + n_va else "test")
    return assignment


def has_tumor(mask_path):
    """True nếu mask có ít nhất 1 pixel khác 0 (để thống kê ảnh có/không u)."""
    try:
        from PIL import Image
        import numpy as np
        return bool(np.array(Image.open(mask_path)).any())
    except Exception:
        return None


def prepare_dirs(dst, force):
    existing = [dst / s for s in SPLITS if (dst / s).exists()]
    if existing and not force:
        sys.exit("[LỖI] Đã tồn tại: " + ", ".join(str(p.relative_to(REPO_ROOT)
                 if REPO_ROOT in p.parents else p) for p in existing) +
                 "\n      Dùng --force nếu muốn xóa và chia lại.")
    for p in existing:
        shutil.rmtree(p)
    for s in SPLITS:
        (dst / s / "images").mkdir(parents=True, exist_ok=True)
        (dst / s / "masks").mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    if any(r <= 0 for r in args.ratio):
        sys.exit("[LỖI] Tỷ lệ phải là số dương, vd: --ratio 0.7 0.15 0.15")

    src, dst = resolve(args.src), resolve(args.dst)
    pairs = find_pairs(src)
    n_patients = len({p for p, _, _ in pairs})
    print(f"Tìm thấy {len(pairs)} cặp ảnh/mask từ {n_patients} bệnh nhân.")

    assignment = assign_splits(pairs, args.ratio, args.seed, args.by)
    prepare_dirs(dst, args.force)

    op = shutil.move if args.move else shutil.copy2
    rows = []
    stats = {s: {"images": 0, "patients": set(), "tumor": 0} for s in SPLITS}

    for idx, (patient, img, mask) in enumerate(pairs):
        split = assignment[idx]
        new_img = dst / split / "images" / img.name
        new_mask = dst / split / "masks" / mask.name
        tumor = has_tumor(mask)          # đọc mask trước khi có thể bị move
        op(str(img), str(new_img))
        op(str(mask), str(new_mask))
        stats[split]["images"] += 1
        stats[split]["patients"].add(patient)
        stats[split]["tumor"] += 1 if tumor else 0
        rows.append([img.name, mask.name, patient, split,
                     "" if tumor is None else int(tumor)])

    # Ghi manifest: cả nhóm dùng để tái lập đúng cách chia
    manifest = dst / "splits.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "mask", "patient", "split", "has_tumor"])
        w.writerows(sorted(rows, key=lambda r: (r[3], r[0])))

    # Kiểm tra: không được trùng giữa các tập
    names = {s: {r[0] for r in rows if r[3] == s} for s in SPLITS}
    assert not (names["train"] & names["val"] or names["train"] & names["test"]
                or names["val"] & names["test"]), "Có ảnh bị trùng giữa các tập!"
    if args.by == "patient":
        pts = {s: stats[s]["patients"] for s in SPLITS}
        assert not (pts["train"] & pts["val"] or pts["train"] & pts["test"]
                    or pts["val"] & pts["test"]), "Có bệnh nhân bị trùng giữa các tập!"

    # Báo cáo
    total = len(pairs)
    print(f"\nChia theo: {args.by} | seed={args.seed} | "
          f"tỷ lệ={'/'.join(f'{r:g}' for r in args.ratio)}")
    print(f"{'Tập':<7}{'Ảnh':>7}{'Tỷ lệ':>9}{'Bệnh nhân':>12}{'Ảnh có u':>11}")
    for s in SPLITS:
        st = stats[s]
        print(f"{s:<7}{st['images']:>7}{st['images']/total:>8.1%}"
              f"{len(st['patients']):>12}{st['tumor']:>11}")
    print(f"\nĐã ghi: {manifest}")

    if args.zip:
        zip_path = dst / "dataset_split.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for s in SPLITS:
                for f in sorted((dst / s).rglob("*")):
                    if f.is_file():
                        z.write(f, f.relative_to(dst))
            z.write(manifest, manifest.name)
        print(f"Đã nén: {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
