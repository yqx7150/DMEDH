# DMEDH

**Paper**: DMEDH: Diffusion Model-boosted Multiplane Extrapolation for Digital Holographic Reconstruction

**Authors**: Songyang Gao, Weisheng Xu, Xinyi Wu, Jiawei Liu, Bofei Wang, Tianya Wu, Wenbo Wan*, and Qiegen Liu*   

Optics Express, https://doi.org/10.1364/OE.531147      

Date : Aug-22-2024  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2020, School of Information Engineering, Nanchang University.  
# Abstract
Digital holography can reconstruct the amplitude and phase information of the target light field. However, the reconstruction quality is largely limited by the size of the hologram. Multi-plane holograms can impose constraints for reconstruction, yet the quality of the reconstructed images continues to be restricted owing to the deficiency of effective prior information constraints. To attain high-quality image reconstruction, a diffusion model-boosted multiplane extrapolation for digital holographic reconstruction (DMEDH) algorithm is proposed. The dual-channel prior information of amplitude and phase extracted through denoising score matching is employed to constrain the physically driven dual-domain rotational iterative process. Depending on the utilization of multi-plane hologram data, the serial DMEDH and the parallel DMEDH are presented. Compared with traditional methods, simulative and experimental results demonstrate that images reconstructed using DMEDH exhibit better reconstruction quality and have higher structural similarity, peak signal-to-noise ratios, and strong generalization. The reconstructed image using DMEDH from two holograms exhibits better quality than that of traditional methods from five holograms.


# Main procedure and performance
![DMEDH-s](/Figures/fig1.png "Main procedure and performance")

![DMEDH-p](/Figures/fig2.png "Main procedure and performance")

![Resolution target](/Figures/fig4.png "Main procedure and performance")

![Film](/Figures/fig5.png "Main procedure and performance")
# Environment
```
docker's environment:(cuda10.2,ubuntu16.04) 

docker pull zieghart/base:C10U16_perfect 

conda activate ncsn
```

## Optical system configuration.
![DMEDH-p](/Figures/fig3.png "Main procedure and performance")

## Checkpoints
We provide the pre-trained model. Click [pre-trained model]( https://pan.baidu.com/s/1bJu6ererzLQgiSEkou9H6A?pwd=DMED ) to download the pre-trained model.(Extraction code: DMED)

## Dataset
Please refer to the methods in the paper to create the dataset, and save individual data as .mat files. Organize the dataset into the following structure:

```
data
  train
    amp
    phase
  test
    amp
    phase
```
 
## Training
  Before start to training, the config file needs modifiction. The config path is `/datasets.py`.

  Once you have modified the config file, run the following code to train your own model

  `python main.py --config=aapm_sin_ncsnpp_gb.py --workdir=exp --mode=train --eval_folder=result`

## Reconstruction

```
python A_1k_arg_PCsampling_demo.py --planes=5 --gpu=0 --useNet=True

python A_1k_arg_PCsampling_demo.py --planes=3 --gpu=0 --useNet=True
```
In the file `A_1k_arg_PCsampling_demo.py`, you can modify the method used by changing these two lines. The method `A_1k_arg_sampling_exper_sim_DMEDH_s` can be replaced with other methods.

```
import A_1k_arg_sampling_exper_sim_DMEDH_s as sampling
from A_1k_arg_sampling_exper_sim_DMEDH_s import ReverseDiffusionPredictor,LangevinCorrector,AnnealedLangevinDynamics ,EulerMaruyamaPredictor,AncestralSamplingPredictor
```

## Acknowledgement
  Thanks to these repositories for providing us with method code and experimental data: https://github.com/THUHoloLab/MPEPI , https://github.com/yqx7150/HoloDiffusion

## Other Related Projects
  * Lens-less imaging via score-based generative model  
[<font size=5>**[Paper]**</font>](https://www.opticsjournal.net/M/Articles/OJf1842c2819a4fa2e/Abstract)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/LSGM)

  * Multi-phase FZA Lensless Imaging via Diffusion Model  
[<font size=5>**[Paper]**</font>](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-12-20595&id=531211)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MLDM)

  * Dual-domain Mean-reverting Diffusion Model-enhanced Temporal Compressive Coherent Diffraction Imaging  
[<font size=5>**[Paper]**</font>](https://doi.org/10.1364/OE.517567)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DMDTC)  
   
  * Imaging through scattering media via generative diffusion model  
[<font size=5>**[Paper]**</font>](https://doi.org/10.1063/5.0180176)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/ISDM)

  * High-resolution iterative reconstruction at extremely low sampling rate for Fourier single-pixel imaging via diffusion model  
[<font size=5>**[Paper]**</font>](https://doi.org/10.1364/OE.510692)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/FSPI-DM)

  * Real-time intelligent 3D holographic photography for real-world scenarios  
[<font size=5>**[Paper]**</font>](https://doi.org/10.1364/OE.529107)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Intelligent-3D-holography)
