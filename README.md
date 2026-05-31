# SignPoseVAE
Official implementation of Sign Pose VAEs from the paper "The Impact of VAE Design on Latent Pose Representations for Diffusion-basedSign Language Production" (CVPRW GenSign 2026).

<p align="center">
  <img src="./docs/vae_gensign.png" width="800"/>
  <b>Architecture of the Sign Pose VAE.</b>
</p>

## 1. Overview

This repository provides scripts and tools to:
- easily define multiple variants of
Variational Autoencoders (VAEs) to encode skeletal pose sequences
- train them
- evaluate them and characterize their latent space through different metrics

It is typically made to be used in the context of sign language processing
tasks and in particular for sign language production, e.g. when using 
a latent diffusion model (or conditional flow matching on latent space), which
is for what it was built.

#### i. Input data format

Originally, the VAE expects as input 178-key points skeletal data as defined in
https://github.com/walsharry/SLRTP_Skeleton_Keypoint_information.
Other skeletal data formats can be handled modulo some adaptation of the scripts notably
in:
- [data/skeletal_data.py](data/skeletal_data.py) (contains the definition of regions nodes and graph connections)
- [model/autoencoders/skelmotionmultivae.py](model/autoencoders/skelmotionmultivae.py)

#### ii. General VAE design

The VAE is defined through the `SkelMotionMultiRegionVAE` class which separates
the input in 4 regions: the torso + arms, the right hand, the left hand and
the face. Note that for the face, inputs are transformed by the model before
encoding to landmarks coordinates displacements at each frame from
a "neutral" (mean) face template.

Each region inputs are processed by independent encoder and decoder modules whose architecture
can be modified in the configuration file under the `model` keyword (cf. configuration examples
in the [configs/](configs) folder. However, one can either choose to
predict the latent distribution parameters $\mu$ and $\sigma^2$ from a concatenation
of the different encoders outputs (setting the `shared_latent_distribution` parameter to
`True`) or to predict independent normal distributions parameters per-region 
(`shared_latent_distribution=False`).

Please refer to the scheme and/or the paper for more details.

#### iii. Training loss design

To train the VAE, we adopt the approach of 
[I. Higgins et al. *beta-VAE* 2017 paper](https://openreview.net/pdf?id=Sy2fzU9gl) by
minimizing the following loss:

$$\mathcal{L}_{\text{VAE}}:=\mathcal{L}_{\text{recon}} + \beta~ D_\text{KL} \left( \mathcal{N}(\mu, \sigma^2 \textbf{I}) || \mathcal{N}(0, \textbf{I}) \right)$$

The base **reconstruction loss** is a $\ell_1$ error loss (MAE) with the same scaling factor for each
region error but can be changed in the configuration file under the `losses: recon` key-words.
For instance, the following configuration:
```yaml
losses:
  recon:
    list_losses:
      - torsoarms_position
      - rh_position
      - lh_position
      - face_position
      - torsoarms_velocity
    scaling_factors:
      torsoarms_position: 10.
      rh_position: 20.
      lh_position: 15.
      face_position: 5.
      torsoarms_velocity: 7.5
    losses_params:
      torsoarms_position:
        loss: mse
      rh_position:
        loss: l1
      lh_position:
        loss: l1
      face_position:
        loss: l1
      torsoarms_velocity:
        loss: mse
```

is equivalent to

$$\mathcal{L}_{\text{recon}} := ???$$

The **Kullback-Leibler divergence loss** ???

#### iv. The VAE variants

Although one can define its own personalized configuration for 
encoder-decoder architectures, regions' latent dimensions and distribution,
we provide the curated configurations of the 4 variants studied in our paper
in the [configs/](configs) folder. The following table is a recap of the
main differences between variants:

<p align="center">
  <img src="./docs/variants.png" width="700"/>
  <b>Sign Pose VAE variants.</b>
</p>

## 2. Setup

## 3. Usage

NB: before using latent pose representations to train a latent generative models,
we recommend standardizing the latent poses as ??? (and de-standardize the generated outputs before
decoding).

## 4. Outputs examples

---

### Citation

If you use this repository for your research, please cite it as follows:
```text
@InProceedings{Faure_2026_CVPR,
    author    = {Faur\'e, Guilhem and Sadeghi, Mostafa and Bigeard, Sam and Ouni, Slim},
    title     = {The Impact of VAE Design on Latent Pose Representations for Diffusion-based Sign Language Production},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {10631-10640}
}
```