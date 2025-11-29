import os
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Get csv of image paths by position from MIMIC-CXR dataset"
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./dataset/mimic-cxr-dataset",
        help="Path to the root of the MIMIC-CXR dataset",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default="./dataset/mimic-cxr-dataset/official_data_iccv_final/files",
        help="Path to the images folder",
    )
    parser.add_argument(
        "--meta_root",
        type=str,
        default="./dataset/mimic-cxr-dataset/metadata.csv",
        help="Path to the metadata csv file",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="PA",
        help="View position to filter images by. One of 'PA', 'AP', 'LATERAL'",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Path to save the output csv file with image paths",
    )
    # NEW: train fraction (rest will be test)
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (rest is test)",
    )

    return parser.parse_args()

def get_image_paths(images_root, meta_df, position="PA"):
    if "dicom_id" not in meta_df.columns or "ViewPosition" not in meta_df.columns:
        raise Exception("Metadata dataframe must contain 'dicom_id' and 'ViewPosition' columns")

    img_paths = []
    not_founds = 0

    for part in os.listdir(images_root):
        part_path = os.path.join(images_root, part)  # /files/p10
        if not os.path.isdir(part_path):
            continue

        for pn in os.listdir(part_path):
            pn_path = os.path.join(part_path, pn)  # /files/p10/p10000032
            if not os.path.isdir(pn_path):
                continue

            for sn in os.listdir(pn_path):
                sn_path = os.path.join(pn_path, sn)  # /files/p10/p10000032/s50414267
                if not os.path.isdir(sn_path):
                    continue

                for jpg in os.listdir(sn_path):
                    if not jpg.lower().endswith(".jpg"):
                        continue

                    jpg_folder = sn_path
                    jpg_path = os.path.join(sn_path, jpg)
                    jpg_name = os.path.splitext(jpg)[0]

                    img_meta = meta_df[meta_df["dicom_id"] == jpg_name]
                    if img_meta.empty:
                        not_founds += 1
                        continue

                    img_position = img_meta.iloc[0]["ViewPosition"]
                    if img_position == position:
                        img_paths.append(
                            {
                                "img_path": jpg_path,
                                "folder_path": jpg_folder,
                                "dicom_id": jpg_name,
                                "view_position": img_position,
                            }
                        )

    if not_founds > 0:
        print(f"Total images not found in metadata: {not_founds}")

    return pd.DataFrame(img_paths)

def make_data(dataset_root, images_root, meta_root, position="PA"):
    try:
        meta_df = pd.read_csv(meta_root)
    except Exception as e:
        raise Exception(f"Error reading metadata file: {meta_root}") from e

    img_paths_df = get_image_paths(images_root, meta_df, position=position)
    return img_paths_df

if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.dataset_root):
        raise Exception(f"Dataset folder not found: {args.dataset_root}")

    if not os.path.exists(args.images_root):
        # try a default inside dataset_root
        default_images_root = os.path.join(args.dataset_root, "official_data_iccv_final", "files")
        if not os.path.exists(default_images_root):
            raise Exception(f"Images folder not found. Tried: {args.images_root} and {default_images_root}")
        args.images_root = default_images_root

    if not os.path.exists(args.meta_root):
        default_meta = os.path.join(args.dataset_root, "metadata.csv")
        if not os.path.exists(default_meta):
            raise Exception(f"Metadata file not found. Tried: {args.meta_root} and {default_meta}")
        args.meta_root = default_meta

    if args.position not in ["PA", "AP", "LATERAL"]:
        raise Exception(f"Invalid position: {args.position}. Must be one of 'PA', 'AP', 'LATERAL'")

    if not (0.0 < args.train_frac < 1.0): 
        raise Exception(f"train_frac must be between 0 and 1, got {args.train_frac}")

    img_paths_df = make_data(
        dataset_root=args.dataset_root,
        images_root=args.images_root,
        meta_root=args.meta_root,
        position=args.position,
    )

    if img_paths_df.empty:
        raise Exception(f"No images found for position {args.position}")

    # shuffle and split into train/test
    img_paths_df = img_paths_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n_total = len(img_paths_df)
    n_train = int(n_total * args.train_frac)
    if n_train == 0 or n_train == n_total:
        raise Exception(
            f"train_frac={args.train_frac} leads to empty train or test split "
            f"(n_total={n_total}, n_train={n_train}). Adjust train_frac."
        )

    train_df = img_paths_df.iloc[:n_train].copy()
    test_df = img_paths_df.iloc[n_train:].copy()

    if args.output_csv == "":
        args.output_csv = os.path.join(args.dataset_root, f"image_paths_{args.position}.csv")

    img_paths_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(img_paths_df)} rows (full) to {args.output_csv}")

    # also save train and test CSVs
    base, ext = os.path.splitext(args.output_csv)
    train_csv = base + "_train" + ext
    test_csv = base + "_test" + ext

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Saved {len(train_df)} train rows to {train_csv}")
    print(f"Saved {len(test_df)} test rows to {test_csv}")
