import os
import numpy as np
import glob


# ============================================================
# Dataset root
# ============================================================

DATASET_ROOT = "/home/aashishbishow/ProjectX/Moonbeam Multi-Task Data"


def scan_dataset(folder_name, folder_path):

    # Recursively find ALL .npy files
    files = glob.glob(
        os.path.join(folder_path, "**", "*.npy"),
        recursive=True
    )

    print(f"\n{'=' * 70}")
    print(f"{folder_name}")
    print(f"{'=' * 70}")
    print(f"Searching: {folder_path}")
    print(f"Total .npy files found recursively: {len(files):,}")

    # Remove auxiliary files
    main_files = [
        f for f in files
        if "bar_beat_chord" not in os.path.basename(f)
        and not os.path.basename(f).endswith("_chord.npy")
    ]

    print(f"Main sequence files: {len(main_files):,}")

    if not main_files:
        print("\nNo main sequence files found!")

        if files:
            print("\nExample .npy files found:")
            for f in files[:10]:
                print(f"  {f}")
        else:
            print("\nNO .npy FILES FOUND AT ALL.")

            if os.path.exists(folder_path):
                print("Directory contents:")
                for item in os.listdir(folder_path)[:20]:
                    print(f"  {item}")
            else:
                print("PATH DOES NOT EXIST:")
                print(f"  {folder_path}")

        print("-" * 70)

        return 0

    # ========================================================
    # Statistics
    # ========================================================

    max_file = ""
    max_len = 0

    min_file = ""
    min_len = float("inf")

    total_tokens = 0

    over_1024 = 0

    valid_files = 0
    unreadable_files = 0

    # ========================================================
    # Scan files
    # ========================================================

    for f in main_files:

        try:

            # IMPORTANT:
            #
            # Do NOT use mmap_mode="r" here.
            #
            # Some SLAKH files are object arrays and require
            # allow_pickle=True.
            #
            arr = np.load(
                f,
                allow_pickle=True
            )

            length = arr.shape[0]

        except Exception as e:

            unreadable_files += 1

            print(f"Skipping: {f}")
            print(f"Reason: {e}")

            continue

        valid_files += 1

        # ====================================================
        # Total tokens
        # ====================================================

        total_tokens += length

        # ====================================================
        # Maximum
        # ====================================================

        if length > max_len:

            max_len = length
            max_file = f

        # ====================================================
        # Minimum
        # ====================================================

        if length < min_len:

            min_len = length
            min_file = f

        # ====================================================
        # Context limit
        # ====================================================

        if length > 1024:
            over_1024 += 1

    # ========================================================
    # Statistics
    # ========================================================

    if valid_files > 0:

        average_tokens = (
            total_tokens / valid_files
        )

    else:

        average_tokens = 0

    print()

    print(
        f"Valid files:             "
        f"{valid_files:,}"
    )

    print(
        f"Unreadable files:        "
        f"{unreadable_files:,}"
    )

    print()

    print(
        f"TOTAL TOKENS:            "
        f"{total_tokens:,}"
    )

    print(
        f"Average tokens/file:     "
        f"{average_tokens:,.2f}"
    )

    print()

    if valid_files > 0:

        print(
            f"Shortest file:           "
            f"{os.path.basename(min_file)} "
            f"({min_len:,} tokens)"
        )

        print(
            f"Longest file:            "
            f"{os.path.basename(max_file)} "
            f"({max_len:,} tokens)"
        )

    print()

    print(
        f"Files > 1024 tokens:     "
        f"{over_1024:,}"
    )

    print()

    print("Example files:")

    for f in main_files[:5]:
        print(f"  {f}")

    print("-" * 70)

    return total_tokens


# ============================================================
# DATASETS
# ============================================================

commu_tokens = scan_dataset(
    "ComMU",
    os.path.join(
        DATASET_ROOT,
        "ComMU",
        "processed"
    )
)


slakh_tokens = scan_dataset(
    "SLAKH2100",
    os.path.join(
        DATASET_ROOT,
        "SLAKH2100",
        "processed"
    )
)


emopia_tokens = scan_dataset(
    "EMOPIA2.2",
    os.path.join(
        DATASET_ROOT,
        "EMOPIA2.2",
        "processed"
    )
)


# ============================================================
# GRAND TOTAL
# ============================================================

total_all_datasets = (
    commu_tokens
    + slakh_tokens
    + emopia_tokens
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print("=" * 70)
print("MOONBEAM MULTI-TASK DATASET TOKEN SUMMARY")
print("=" * 70)

print(
    f"ComMU:              "
    f"{commu_tokens:,}"
)

print(
    f"SLAKH2100:          "
    f"{slakh_tokens:,}"
)

print(
    f"EMOPIA2.2:          "
    f"{emopia_tokens:,}"
)

print("-" * 70)

print(
    f"ALL DATASETS:       "
    f"{total_all_datasets:,} TOKENS"
)

print("=" * 70)