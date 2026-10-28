"""Interactive annotation tool. Saves after every answer and resumes where you stopped.

Usage:
  python -m scripts.annotate --file data/labels/topic_validation.csv --column fits \
      --choices y,n --show title,label,snippet
"""
import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--column", required=True)
    p.add_argument("--choices", required=True, help="comma separated, e.g. y,n")
    p.add_argument("--show", required=True, help="comma separated columns to display")
    args = p.parse_args()

    choices = [c.strip() for c in args.choices.split(",")]
    show = [c.strip() for c in args.show.split(",")]
    df = pd.read_csv(args.file, keep_default_na=False, dtype=str)
    todo = df.index[df[args.column].str.strip() == ""].tolist()
    total = len(df)
    print(f"{total - len(todo)}/{total} already done. Choices: {choices}  (q = save & quit, s = skip)\n")

    for idx in todo:
        print("-" * 70)
        print(f"[{idx + 1}/{total}]")
        for col in show:
            print(f"{col.upper()}: {df.at[idx, col]}")
        while True:
            try:
                ans = input(f"-> {'/'.join(choices)} : ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "q"
            if ans in choices:
                df.at[idx, args.column] = ans
                df.to_csv(args.file, index=False)
                break
            if ans == "s":
                break
            if ans == "q":
                df.to_csv(args.file, index=False)
                done = (df[args.column].str.strip() != "").sum()
                print(f"\nSaved. {done}/{total} annotated.")
                return
            print("   invalid, try again")

    df.to_csv(args.file, index=False)
    print("\nAll done. Saved.")


if __name__ == "__main__":
    main()
