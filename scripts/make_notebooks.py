"""Build the three Colab notebooks from one definition so they never drift apart.

    python scripts/make_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]

INTRO = {
    "smoke": (
        "# Mode A: Smoke test\n\n"
        "Checks that every stage runs on a real GPU with two tiny models (Qwen2.5-0.5B and 1.5B) on 24 "
        "questions. **Its numbers mean nothing and must never be reported.** Run this first after any change.\n\n"
        "Runtime: about 10 to 20 minutes on a free T4, most of it installation."
    ),
    "pilot": (
        "# Mode B: Pilot experiment\n\n"
        "Runs the real models on 200 questions (100 TruthfulQA, 50 TriviaQA, 50 SciQ). Use it to check three "
        "things before the full run: the number of consistently wrong (CW) items, whether CPS shows a signal "
        "(AUROC above 0.6), and the runtime per stage. The go or no go rule is in COLAB_EXECUTION_GUIDE.md."
    ),
    "full": (
        "# Mode C: Full experiment\n\n"
        "Runs every stage for both models, then writes the results into the paper. Each stage saves its output "
        "to Google Drive and skips work already done, so if Colab disconnects, reconnect and run the cells again "
        "from the top. Stages run one cell at a time so a session limit only loses the current stage."
    ),
}


def cells(mode: str) -> list:
    config = f"configs/{mode}.yaml"
    out = [new_markdown_cell(INTRO[mode])]
    out.append(new_markdown_cell("## 1. Settings\nEdit `REPO_URL` once the repository is on GitHub. "
                                 "Keep `USE_DRIVE = True` so results survive a disconnect."))
    out.append(new_code_cell(
        f'MODE = "{mode}"\n'
        f'CONFIG = "{config}"\n'
        'REPO_URL = "https://github.com/YOUR-ACCOUNT/agreement-is-not-truth.git"  # or upload the ZIP to Drive\n'
        'ZIP_ON_DRIVE = "/content/drive/MyDrive/agreement-is-not-truth.zip"     # used when REPO_URL is not set\n'
        'USE_DRIVE = True\n'
        'INSTALL_VLLM = False  # True on L4/A100 for speed; the code falls back to transformers if vLLM fails\n'
        'PROJECT = "/content/agreement-is-not-truth"'))
    out.append(new_markdown_cell("## 2. Hardware check"))
    out.append(new_code_cell("!nvidia-smi --query-gpu=name,memory.total --format=csv"))
    out.append(new_markdown_cell("## 3. Get the code and connect Drive"))
    out.append(new_code_cell(
        "import os, subprocess\n"
        "if USE_DRIVE:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "if not os.path.exists(PROJECT):\n"
        "    if 'YOUR-ACCOUNT' not in REPO_URL:\n"
        "        subprocess.run(['git', 'clone', REPO_URL, PROJECT], check=True)\n"
        "    else:\n"
        "        subprocess.run(['unzip', '-q', ZIP_ON_DRIVE, '-d', '/content'], check=True)\n"
        "os.chdir(PROJECT)\n"
        "if USE_DRIVE:\n"
        "    store = '/content/drive/MyDrive/agreement-is-not-truth-results'\n"
        "    os.makedirs(store, exist_ok=True)\n"
        "    if not os.path.islink('results'):\n"
        "        subprocess.run(['rm', '-rf', 'results'])\n"
        "        os.symlink(store, 'results')\n"
        "print(os.getcwd(), os.listdir('.'))"))
    out.append(new_markdown_cell("## 4. Install dependencies"))
    out.append(new_code_cell(
        "!pip install -q -e .[gpu,judge,dev]\n"
        "if INSTALL_VLLM:\n"
        "    !pip install -q vllm"))
    out.append(new_markdown_cell("## 5. Keys\nAdd `HF_TOKEN` and (optional) `OPENAI_API_KEY` in Colab's "
                                 "Secrets panel (key icon on the left). Without an OpenAI key the judge falls "
                                 "back to alias matching plus NLI, and the paper reports which judge was used."))
    out.append(new_code_cell(
        "import os\n"
        "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # less GPU memory fragmentation\n"
        "try:\n"
        "    from google.colab import userdata\n"
        "    for name in ('HF_TOKEN', 'OPENAI_API_KEY'):\n"
        "        try:\n"
        "            os.environ[name] = userdata.get(name)\n"
        "            print(name, 'loaded')\n"
        "        except Exception:\n"
        "            print(name, 'not set')\n"
        "except ImportError:\n"
        "    print('Not on Colab; set keys in the environment')"))
    if mode == "smoke":
        out.append(new_markdown_cell("## 6. Offline tests\nRuns the unit and integration tests on the simulated "
                                     "backend. They need no GPU and should all pass."))
        out.append(new_code_cell("!python -m pytest -q tests"))
    out.append(new_markdown_cell("## 7. Run the experiment\nEach cell is one stage. Rerunning a cell skips "
                                 "finished work, so it is safe after a disconnect."))
    stages = ["prepare", "generate", "judge", "sample_scores", "split", "probe", "metaqa", "probe_scores",
              "evaluate"]
    for stage in stages:
        out.append(new_code_cell(f"!python -m pressure_probe.cli stage {stage} --config $CONFIG"))
    if mode == "pilot":
        out.append(new_markdown_cell("## Go or no go\nContinue to the full run only if every model has at "
                                     "least 15 CW answers and CPS AUROC is above 0.6 for at least one model."))
        out.append(new_code_cell(
            "import json\n"
            "summary = json.load(open('results/pilot/summary.json'))\n"
            "for key, model in summary['models'].items():\n"
            "    cw = model['counts']['CW']\n"
            "    cps = model['methods']['cps']['auroc_matched']['value']\n"
            "    print(f\"{model['display_name']}: CW = {cw}, CPS AUROC = {cps}\")\n"
            "    for dataset, n in model['counts']['per_dataset'].items():\n"
            "        print(f\"   {dataset}: {n['CW']} CW, {n['CR']} CR, {n['INC']} INC out of {n['total']}\")\n"
            "    for stage, info in model.get('timings', {}).items():\n"
            "        print(f\"   {stage}: {info['seconds'] / 60:.1f} min\")"))
    if mode != "smoke":
        out.append(new_markdown_cell(
            "## 8. Human labels (needed for the paper)\nCreates `human_labels.csv` with 100 answers. Download it, "
            "have two authors fill `rater1` and `rater2` independently (1 = correct, 0 = wrong), and put the "
            "filled file back in the same folder. The report then prints Cohen's kappa in the paper."))
        out.append(new_code_cell("!python scripts/make_label_sheet.py --config $CONFIG"))
    out.append(new_markdown_cell("## 9. Install LaTeX and pandoc (about 5 minutes)"))
    out.append(new_code_cell(
        "!apt-get -qq update && apt-get -qq install -y texlive-latex-recommended texlive-latex-extra "
        "texlive-fonts-recommended texlive-science latexmk > /dev/null && echo LaTeX done\n"
        "# Colab's own pandoc is too old for the Word export, so install a current release.\n"
        "!wget -q https://github.com/jgm/pandoc/releases/download/3.1.3/pandoc-3.1.3-1-amd64.deb "
        "-O /tmp/pandoc.deb && dpkg -i /tmp/pandoc.deb > /dev/null && pandoc --version | head -1"))
    out.append(new_markdown_cell("## 10. Write results into the paper\nFills tables, figures and numbers, "
                                 "compiles the IEEE PDF, writes the Word file, checks the page count and "
                                 "builds the ZIP."))
    out.append(new_code_cell("!python -m pressure_probe.cli report --config $CONFIG"))
    out.append(new_code_cell(
        "import json\n"
        "report = json.load(open('paper/validation_report.json'))\n"
        "print(json.dumps({k: v for k, v in report.items() if k != 'unresolved_tokens'}, indent=1))\n"
        "print('Unresolved tokens:', report['unresolved_tokens'])"))
    if mode == "smoke":
        out.append(new_markdown_cell("The smoke paper is for checking the pipeline only. Do not keep it."))
    out.append(new_markdown_cell("## 11. Download"))
    out.append(new_code_cell(
        "from google.colab import files\n"
        "files.download(f'dist/{MODE}_package.zip')\n"
        "files.download('paper/paper.pdf')"))
    return out


def main() -> None:
    names = {"smoke": "smoke_test", "pilot": "pilot_experiment", "full": "full_experiment"}
    for mode, name in names.items():
        notebook = new_notebook(cells=cells(mode))
        notebook.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}}
        nbformat.validate(notebook)
        target = ROOT / "notebooks" / f"{name}.ipynb"
        nbformat.write(notebook, target)
        print("wrote", target)


if __name__ == "__main__":
    main()
