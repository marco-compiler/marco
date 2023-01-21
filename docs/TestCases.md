# Status of significant test cases in MARCO

## ThermalChip
| Name  | Vars | States | Compile Time OMC      | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|-----------------------|--------------------|--------------|---------------|
| ThermalChipSimpleBoundaryODE N=4, M=4, P=4 | 88 | 64 | 0.592s                | 0.491s             | 0.022s       | 0.010s        |
| ThermalChipSimpleBoundaryODE N=40, M=40, P=40 | ~65k | ~64k | 59m 14.056s           | 0.578s             | 2.724s       | 0.313s           |
| ThermalChipSimpleBoundaryODE N=100, M=100, P=100 | ~1M | ~1M | Didn't end within 29h | 1.391s             | N/A          | 4.453s           |

**Status**:
- Simulation parameters:
   - start time: `0`
   - end time: `1`
   - time-step: `0.001`
- Implicit ranges in equations (e.g. `x[:,:] = ...`) have to be manually converted to explicit ones.

| Name  | Vars | States | Compile Time OMC | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|------------------|--------------------|--------------|---------------|
| ThermalChipSimpleBoundaryDAE N=4, M=4, P=4 | 328 | 64 | 0.708s           | 0.277s             | 0.026s       | 0.010s        |
| ThermalChipSimpleBoundaryDAE N=40, M=40, P=40 | ~300k | ~64k | 264m 11.369s     | 11.631s            | 4.790s       | 0.758s        |
| ThermalChipSimpleBoundaryDAE N=100, M=100, P=100 | ~5M | ~1M | N/A              | 45m 27.086s        | N/A          | 11.585s           |

**Status**:
- Simulation parameters:
    - start time: `0`
    - end time: `1`
    - time-step: `0.001`
- Implicit ranges in equations (e.g. `x[:,:] = ...`) have to be manually converted to explicit ones.

| Name  | Vars | States | Compile Time OMC     | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|----------------------| -------------------|--------------|---------------|
| ThermalChipSimpleBoundaryOO N=4, M=4, P=4 |~1k | 64 | 2.381s               | 0.390s | 0.082s       | 0.012s        |
| ThermalChipSimpleBoundaryOO N=40, M=40, P=40 | ~1M | ~64k | OOM after 5m 44.196s | 0.492s | N/A          | 2.696s           |
| ThermalChipSimpleBoundaryOO N=100, M=100, P=100 | ~16M | ~1M | N/A                  | 1.108s | N/A          | 28.640s           |

**Status**:
- Simulation parameters:
    - start time: `0`
    - end time: `1`
    - time-step: `0.001`

## ThermalChipCooling

| Name  | Vars | States | Compile Time OMC | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|------------------|--------------------|--------------|---------------|
| ThermalChipCoolingSimpleBoundaryODE N=4, M=4, P=4 | 145 | 80 | 2.710s           | 1.035s             | 0.032s       | 0.011s        |
| ThermalChipCoolingSimpleBoundaryODE N=40, M=40, P=40 | ~145k | ~65k | 95m 38.550s      | 1.175s             | 1.785s       | 0.400s           |
| ThermalChipCoolingSimpleBoundaryODE N=100, M=100, P=100 | ~2M | ~1M | N/A              | 6.712s                | N/A          | 5.585s           |

**Status**:
- Simulation parameters:
    - start time: `0`
    - end time: `1`
    - time-step: `0.001`
- Implicit ranges in equations (e.g. `x[:,:] = ...`) have to be manually converted to explicit ones.

| Name  | Vars | States | Compile Time OMC        | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|-------------------------| -------------------|--------------|---------------|
| ThermalChipCoolingSimpleBoundaryOO N=4, M=4, P=4 | ~1K | 80 | 2.649s                  | 0.478s | 0.061s       | 0.013s        |
| ThermalChipCoolingSimpleBoundaryOO N=40, M=40, P=40 | ~500k | ~65k | Compilation failing (*) | 0.651s | N/A          | 2.589s           |
| ThermalChipCoolingSimpleBoundaryOO N=100, M=100, P=100 | ~8M | ~1M | N/A                     | 1.545s | N/A          | 29.126s           |

**Status**:
- Simulation parameters:
    - start time: `0`
    - end time: `1`
    - time-step: `0.001`
- Implicit ranges in equations (e.g. `x[:,:] = ...`) have to be manually converted to explicit ones.

(*) Log from OMC:
```
[ThermalChipCoolingOO.mo:45:7-45:35:writable] Error: Model is structurally singular, error found sorting equations
  283824: vol[24,33,38].upper.Q = 0.0296 * (vol[23,33,38].lower.T - vol[24,33,38].T)
  261563: vol[24,33,38].upper.Q = 0.0296 * (vol[23,33,38].lower.T - vol[24,33,38].T)
  261564: -vol[24,33,38].upper.Q = 0.0296 * (vol[23,33,38].lower.T - vol[23,33,38].T)
for variables
  186878: vol[24,33,38].upper.Q:VARIABLE(flow=true unit = \"W\" )  type: Real [40,40,40]
  198035: vol[23,33,38].lower.T:VARIABLE(flow=false unit = \"K\" nominal = 500.0 )  type: Real [40,40,40]
  164536: vol[26,34,1].T:STATE(1)(start = Tstart unit = \"K\" fixed = true nominal = 500.0 )  \"Volume temperature\" type: Real [40,40,40]
Error: Internal error Transformation Module PFPlusExt index Reduction Method Pantelides failed!
Error: post-optimization module removeSimpleEquations (simulation) failed.
```

## MethanolHeatExchangers

| Name  | Vars | States | Compile Time OMC       | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|------------------------|--------------------|--------------|---------------|
| MethanolHeatExchangers Nu=3, Nh=4, Nv=6 | 546 | 147 | 1.754s                 | 0.290s             | 0.079s       | 0.068s        |
| MethanolHeatExchangers Nu=30, Nh=40, Nv=20 | ~100k | ~24k | 29m 59.637s            | 0.495s             | 17.343s      | 11.919s       |
| MethanolHeatExchangers Nu=300, Nh=40, Nv=20 | ~1M | ~240k | OOM after 229m 25.481s | 1.835s             | N/A          | 1m 19.284s    |

**Status**:
 - Simulation parameters:
   - start time: `0`
   - end time: `10`
   - time-step: `0.01`
 - Implicit ranges in equations (e.g. `x[:,:] = ...`) have to be manually converted to explicit ones.
 - Iterators have to be manually unrolled (e.g. `x[i] for i in 1:40`).

## PowerGrid

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| PowerGridBaseDAE Ne=2 | 156 | 16 | TBD | TBD | TBD | TBD |
| PowerGridBaseDAE Ne=100 | ~400k | ~40k | TBD | TBD | TBD | TBD |
| PowerGridBaseDAE Ne=400 | ~6M | ~640k | TBD | TBD | TBD | TBD |

**Status**: blah blah blah

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| PowerGridSecondaryDAE Ne=2 | 158 | 17 | TBD | TBD | TBD | TBD |
| PowerGridSecondaryDAE Ne=100 | ~400k | ~40k | TBD | TBD | TBD | TBD |
| PowerGridSecondaryDAE Ne=400 | ~6M | ~640k | TBD | TBD | TBD | TBD |

**Status**: blah blah blah

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| PowerGridBaseOO Ne=2 | 364 | 16 | TBD | TBD | TBD | TBD |
| PowerGridBaseOO Ne=100 | ~800k | ~40k | TBD | TBD | TBD | TBD |
| PowerGridBaseOO Ne=400 | ~12M | ~600k | TBD | TBD | TBD | TBD |

**Status**: blah blah blah

## ScalableTestGrids

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| Type1 N=1, M=1  | ~2k | 4 | TBD | TBD | TBD | TBD |
| Type1 N=16, M=4 | ~800k | ~1k | TBD | TBD | TBD | TBD |
| Type1 N=45, M=4 | ~6.4M | ~8k | TBD | TBD | TBD | TBD |

**Status**: Currently trying to get proper flat Modelica output with non-scalarized bindings. All known issues have been fixed so far.
The model contains conditional expressions for saturation blocks, so it is necessary to create a variant using smooth saturation functions
to handle them in MARCO until we have proper event handling.
The model includes a non-trivial initialization problem using homotopy, we should discuss to which extent we can support that.
