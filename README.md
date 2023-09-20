# Linear-FEM

Prerequisite: 
``` bash
pip3 install taichi
```


## A Minimal Linear-FEM Demo

``` bash
python3 linear_fem_minimal.py
```

## Invertible Constitutive Models

``` bash
python3 linear_fem_invertible.py
```

This script includes the implementation of
1. `Invertible corotated` constitutive model. From the paper "Energetically Consistent Invertible Elasticity".
2. `Invertible neo-hookean` constitutive model. From the paper "Stable Neo-Hookean Flesh Simulation".

This script uses the `invertible neo-hookean` model by default. Modify `CONSTITUTIVE_MODEL=INVERTIBLE_COROTATED` in code to use the `invertible coroated` model.

The results of the two models are shown as follows:
|Invertible Corotated | Invertible Neo-Hookean| 
|          ---        |         ---           |
|  ![video_coro](https://github.com/YuCrazing/Linear-FEM/assets/8120108/9118378f-2cf8-47d2-bda7-4c3c50f30484)  | ![video_neo](https://github.com/YuCrazing/Linear-FEM/assets/8120108/5402974c-6d70-4304-a2f8-4596bd3dd547) |

### A Corrected Derivation of Invertible Neo-Hookean Model

The SIGGRAPH course "Dynamic Deformables: Implementation and Production Practicalities" is an excellent article. However, there are some typos in the derivation of the invertible neo-hookean model, which could lead to misunderstandings. We have tried to correct these mistakes, and you can find more details here: [A Derivation of Stable Neo-Hookean Constitutive Model](https://yucrazing.github.io/assets/files/Stable_NeoHookean.pdf).
