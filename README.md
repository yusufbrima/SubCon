# Implementation for the Sub-Space Contrastive Representation Learning 

## Sample Images from 3DShapes

![Alt Text](Figures/output_figure.png)

### Conditioning on  shape type generative factor

![Alt Text](Figures/output.gif)

### Conditioning on floor hue

![Alt Text](Figures/floor_hue.gif)

## TNSE Visualization of training of MNIST with Triplet and Supervised Contrastive Learning Objectives

<figure style="display: flex; justify-content: space-between;">
        <img src="Figures/mnist_tripplet_animation_advanced.gif" alt="Image 1" style="width: 40%; max-width: 100%; height: auto;">
        <!-- <figcaption style="width: 45%; text-align: center; font-style: italic;">Training for triplet loss for MNIST</figcaption> -->
        <img src="Figures/mnist_scl_animation.gif" alt="Image 2" style="width: 40%; max-width: 100%; height: auto;">
        <!-- <figcaption style="width: 45%; text-align: center; font-style: italic;">Training for Supervised Contrastive loss for MNIST</figcaption> -->
</figure>


## Creating a Conda Environment from a YAML File

In this guide, we'll walk you through the process of creating a Conda environment from a YAML file. This is useful for sharing or recreating environments with specific package dependencies.

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) is installed on your system.

#### Steps

1. **Activate Conda**: Open your terminal and activate Conda if it's not already active. Replace `your_environment_name` with the desired environment name.

   ```shell
   conda activate your_environment_name
   ```

2. **Activate the New Environment**: Activate the newly created environment.
    ```shell
    conda activate new_environment_name
    ```
3. **Verify Installation**: You can verify that the environment was created successfully by checking its list of installed packages.
    ```shell
    conda list
    ```