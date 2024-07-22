# Environment Setup

- We use conda to install required packages:
    ```bash
        conda env create -f env/equiformer_v2_torch1_ocp2.yml
    ```
    This will create a new environment called `equiformer_v2_torch1_ocp2`.

- We activate the environment:
    ```bash
        export PYTHONNOUSERSITE=True    # prevent using packages from base
        conda activate equiformer_v2_torch1_ocp2
    ```

- Clone OC20 GitHub repository and install `ocpmodels`:
    ```bash
        git clone https://github.com/Open-Catalyst-Project/ocp.git
        cd ocp
        git checkout be0f727a515582b01e6c51672a08f5b693f015e9
        pip install -e .
    ```
    The corresponding version of OCP (FAIR Chemistry) is [here](https://github.com/FAIR-Chem/fairchem/tree/be0f727a515582b01e6c51672a08f5b693f015e9).

- Since we reuse the original codebase of OCP (FAIR Chemistry) and provide a new trainer adding DeNS as an auxiliary task, we need to put this repository under the `ocp` codebase cloned above and rename to `experimental`:
    ```bash
        cd ocp
        git clone https://github.com/atomicarchitects/DeNS experimental
    ```
    In this way, when we launch training under `ocp`, the OCP codebase will look into `experimental` directory and find the new trainer and models contained in this DeNS repository.