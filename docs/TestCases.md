# Status of significant test cases in MARCO

## ThermalChip
| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| ThermalChipODE N=4, M=4, P=4 | TBD | TBD | TBD | TBD | TBD | TBD |
| ThermalChipODE N=16, M=16, P=16 | TBD | TBD | TBD | TBD | TBD | TBD |

(the idea is to have one small and one large test case)

**Status**: blah blah blah

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| ThermalChipDAE N=XX, M=XX, P=XX | TBD | TBD | TBD | TBD | TBD | TBD |
| ThermalChipDAE N=XX, M=XX, P=XX | TBD | TBD | TBD | TBD | TBD | TBD |

**Status**: blah blah blah

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| ThermalChipOO N=XX, M=XX, P=XX | TBD | TBD | TBD | TBD | TBD | TBD |
| ThermalChipOO N=XX, M=XX, P=XX | TBD | TBD | TBD | TBD | TBD | TBD |

**Status**: blah blah blah

## ThermalChipCooling

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| ThermalChipCoolingODE N=XX, M=XX, P=XX | TBD | TBD | TBD | TBD | TBD | TBD |
| ThermalChipCoolingODE N=XX, M=XX, P=XX | TBD | TBD | TBD | TBD | TBD | TBD |

**Status**: blah blah blah

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| ThermalChipOO N=XX, M=XX, P=XX | TBD | TBD | TBD | TBD | TBD | TBD |
| ThermalChipOO N=XX, M=XX, P=XX | TBD | TBD | TBD | TBD | TBD | TBD |

**Status**: blah blah blah

## MethanolHeatExchangers

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| MethanolHeatExchangers Nu=3, Nh=4, Nv=6 | TBD | TBD | TBD | TBD | TBD | TBD |
| MethanolHeatExchangers Nu=30, Nh=40, Nv=20 | TBD | TBD | TBD | TBD | TBD | TBD |

**Status**: blah blah blah

## PowerGrid

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| PowerGridBaseDAE Ne=2 | TBD | TBD | TBD | TBD | TBD | TBD |
| PowerGridBaseDAE Ne=100 | TBD | TBD | TBD | TBD | TBD | TBD |

**Status**: blah blah blah


| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| PowerGridSecondaryDAE Ne=2 | TBD | TBD | TBD | TBD | TBD | TBD |
| PowerGridSecondaryDAE Ne=100 | TBD | TBD | TBD | TBD | TBD | TBD |

**Status**: blah blah blah

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| PowerGridBaseOO Ne=2 | TBD | TBD | TBD | TBD | TBD | TBD |
| PowerGridBaseOO Ne=100 | TBD | TBD | TBD | TBD | TBD | TBD |

**Status**: blah blah blah

## ScalableTestGrids

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| Type1 N=1, M=1  | TBD | TBD | TBD | TBD | TBD | TBD |
| Type1 N=16, M=4 | TBD | TBD | TBD | TBD | TBD | TBD |

**Status**: Currently trying to get proper flat Modelica output with non-scalarized bindings. 
Also, the model contains conditional expressions for saturation blocks, so it is necessary to create a variant using smooth saturation functions
to handle them in MARCO until we have proper event handling.
