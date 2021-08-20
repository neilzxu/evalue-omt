# Code for "Online multiple testing with e-values"
## Setup

Setup tried with [conda 4.8.3](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html) and [python 3.8.3](https://www.python.org/downloads/).
```
conda env create -f environment.yml --name <env name>`
conda activate <env name>
pip install -r requirements.txt
```
We also need DeepPurpose to run real data experiments. In external to this repo, do the following
```
git clone git@github.com:kexinhuang12345/DeepPurpose.git
cd DeepPurpose
pip install -e .
```

To run the numerical simulations in the paper and create the associated plots, run:
`python src/main.py --exp <name of experiment> --processes <# of processes allowed for process pool> --out_dir <directory for saving output figures> --result_dir <directory for saving intermediate outputs`
Possible values of experiment names are `lag_comp` and `wor_comp`.

To run the protein binding prediction experiments do the following.
```
cd src
./run-batch.sh 600 5
```

After that has finished, plot the results with
```
python gather_results.py DPP_results ../figures/wcs
```
