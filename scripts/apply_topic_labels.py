"""Create the final topic labels: start from the LLM suggestion, then apply manual corrections.

Usage: python -m scripts.apply_topic_labels
"""
import pandas as pd

CSV = "docs/topics_overview.csv"

# Manual review: corrections made after reading top_words + sample_titles of each topic.
OVERRIDES = {
    0: "Dates & Years",
    2: "Religion",
    3: "Medicine & Human Body",
    8: "Chemistry & Materials",
    12: "Dates & Years",
    15: "Dates & Years",          # LLM call failed
    17: "Germany & Switzerland",
    18: "Africa",
    19: "Writers & Astronauts",
    20: "Calendar & Time",
    21: "Islands & Volcanoes",
    23: "Caribbean & Latin America",
    24: "Singers & Musicians",
    29: "Internet & Web",
    31: "Comics & Cartoons",
    37: "Cars & Road Transport",
    40: "Nazi Germany & History",
    43: "Dates & Years",
    48: "Singers & Musicians",
    50: "Spain & Spanish World",
    51: "Science & Philosophy",
    53: "Weather & Temperature",
    57: "Dates & Years",
    59: "Dates & Years",
    67: "Water & Rivers",
    68: "Athletes",
    69: "Dates & Years",          # LLM call failed
}


def main():
    df = pd.read_csv(CSV, keep_default_na=False)
    df["label"] = df["label_suggested"]
    for topic, label in OVERRIDES.items():
        df.loc[df["topic"] == topic, "label"] = label

    empty = df[df["label"].astype(str).str.strip() == ""]
    assert empty.empty, f"Topics without a label: {empty['topic'].tolist()}"

    real = df[(df["topic"] != -1) & (df["label_suggested"].astype(str).str.strip() != "")]
    accepted = int((real["label"] == real["label_suggested"]).sum())
    print(f"LLM suggestions accepted as-is: {accepted} / {len(real)} ({accepted / len(real):.0%})")
    print(f"Corrected by hand: {len(real) - accepted}; filled in manually (LLM failed): "
          f"{int((df['label_suggested'].astype(str).str.strip() == '').sum())}")
    print(f"Distinct final labels: {df['label'].nunique()}")

    df.to_csv(CSV, index=False)
    print(f"Saved {CSV}")


if __name__ == "__main__":
    main()
