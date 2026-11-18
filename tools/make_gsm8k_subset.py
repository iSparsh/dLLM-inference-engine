"""Reproduce the frozen GSM8K-100 manifest from the public main/test parquet."""

import argparse
import json
import random
from pathlib import Path
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq

URL = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"


def main(output: Path) -> None:
    with urlopen(URL, timeout=60) as r:
        table = pq.read_table(pa.BufferReader(r.read()))
    if len(table) != 1319:
        raise ValueError("GSM8K split changed; check the dataset revision.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for i in sorted(random.Random(759).sample(range(1319), 100)):
            answer = (
                table["answer"][i].as_py().rsplit("####", 1)[1].strip().replace(",", "")
            )
            f.write(
                json.dumps(
                    {
                        "id": f"gsm8k-{i:04}",
                        "index": i,
                        "question": table["question"][i].as_py(),
                        "answer": answer,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval/gsm8k_100.jsonl"))
    main(parser.parse_args().output)
