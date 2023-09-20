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
1. `Invertible corotated` constitutive model. From paper "Energetically Consistent Invertible Elasticity".
2. `Invertible neo-hookean` constitutive model. From paper "Stable Neo-Hookean Flesh Simulation".

This script uses the `invertible neo-hookean` model by default. Modify `CONSTITUTIVE_MODEL=INVERTIBLE_COROTATED` in code to use the `invertible coroated` model.

The comparison of two models is shown as follows:
|Invertible Corotated | Invertible Neo-Hookean| 
|          ---        |         ---           |
|                     |                       |

### A Corrected Derivation of Invertible Neo-Hookean Model

The SIGGRAPH course "Dynamic Deformables: Implementation and Production Practicalities" is an excellent article. However, there are some typos in the derivation of the invertible neo-hookean model, which could lead to misunderstandings. We have tried to correct these mistakes, and you can find more details here: [A Derivation of Stable Neo-Hookean Constitutive Model](https://yucrazing.github.io/assets/files/Stable_NeoHookean.pdf).