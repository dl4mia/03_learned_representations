import argparse
from pathlib import Path

import jupytext
import nbformat


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("-o", "--output", required=False, default=None)

    return parser


def generate(notebook_file, out_filename=None):
    notebook = jupytext.read(notebook_file, as_version=4, fmt="py:percent")
    # save the solution notebook (py to ipynb)
    with open(Path(notebook_file).stem + ".ipynb", "w") as f:
        nbformat.write(notebook, f)

    # make the exercise notebook
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            tags = cell.metadata.get("tags")
            if tags is not None and "solution" in tags:
                src_lines = cell.source.split("\n")
                new_src = "\n".join([
                    ln for ln in src_lines
                    if ln == "" or ln.startswith("#")
                ])
                cell.source = new_src
                cell.metadata["tags"] = []

    nbformat.validate(notebook)
    if out_filename is None:
        out_filename = Path(notebook_file).stem
        if out_filename.endswith("solution"):
            out_filename = out_filename[:-8]
        out_filename = out_filename + "exercise.ipynb"

    with open(out_filename, "w") as f:
        nbformat.write(notebook, f)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    generate(args.input_file, args.output)
    print("Done!")
