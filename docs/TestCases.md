# Status of significant test cases in MARCO

## ThermalChip
| Name  | Vars | States | Compile Time OMC | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|------------------|--------------------|--------------|---------------|
| ThermalChipSimpleBoundaryODE N=4, M=4, P=4 | 88 | 64 | 0.345s           | 0.491s             | 0.021s       | 0.010s        |
| ThermalChipSimpleBoundaryODE N=40, M=40, P=40 | ~65k | ~64k | 3m13.739s      | 0.578s             | 2.959s       | 0.313s           |
| ThermalChipSimpleBoundaryODE N=100, M=100, P=100 | ~1M | ~1M | Compilation error (*)   | 1.391s             | N/A          | 4.453s           |

**Status**:
- Simulation parameters:
   - start time: `0`
   - end time: `1`
   - time-step: `0.001`
- Implicit ranges in equations (e.g. `x[:,:] = ...`) have to be manually converted to explicit ones.

(*) Log from OMC:
```
Error: Internal error Instantiation of ThermalChipODE.Models.ThermalChipSimpleBoundary failed with no error message.
```

| Name  | Vars | States | Compile Time OMC   | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|--------------------|--------------------|--------------|---------------|
| ThermalChipSimpleBoundaryDAE N=4, M=4, P=4 | 328 | 64 | 0.423s             | 0.277s             | 0.021s       | 0.010s        |
| ThermalChipSimpleBoundaryDAE N=40, M=40, P=40 | ~300k | ~64k | 6m6.130            | 11.631s            | 8.029s       | 0.758s        |
| ThermalChipSimpleBoundaryDAE N=100, M=100, P=100 | ~5M | ~1M | Segmentation fault | 45m 27.086s        | N/A          | 11.585s           |

**Status**:
- Simulation parameters:
    - start time: `0`
    - end time: `1`
    - time-step: `0.001`
- Implicit ranges in equations (e.g. `x[:,:] = ...`) have to be manually converted to explicit ones.

(*) Log from OMC:
```
Limited backtrace at point of segmentation fault
/lib/x86_64-linux-gnu/libc.so.6(+0x43090)[0x14867b577090]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaRuntimeC.so(listAppend+0x7e)[0x14867b43a3f0]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_ResolveLoops_colorNodePartitions+0x3ed)[0x14867c3df076]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_ResolveLoops_partitionBipartiteGraph+0x15a)[0x14867c3df240]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_CommonSubExpression_commonSubExpressionFind+0x54b)[0x14867c4fb67d]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_CommonSubExpression_commonSubExpression+0x14f)[0x14867c4fb92e]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_List_mapFold+0x7f)[0x14867c5c3888]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_BackendDAEUtil_mapEqSystem+0x49)[0x14867c54e4c3]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_BackendDAEUtil_preOptimizeDAE+0x1f8)[0x14867c554815]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_BackendDAEUtil_getSolvedSystem+0x152)[0x14867c555c55]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_SimCodeMain_translateModel+0xa18)[0x14867c0ef7cd]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_CevalScriptBackend_callTranslateModel+0xe4)[0x14867c138332]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_CevalScriptBackend_translateModel+0x27a)[0x14867c12f5af]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_CevalScriptBackend_buildModel+0x433)[0x14867c12ebf0]
/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so(omc_CevalScriptBackend_cevalInteractiveFunctions3+0x6d60)[0x14867c14c0f4]
```

| Name  | Vars | States | Compile Time OMC     | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|----------------------| -------------------|--------------|---------------|
| ThermalChipSimpleBoundaryOO N=4, M=4, P=4 |~1k | 64 | 1.367s               | 0.390s | 0.063s       | 0.012s        |
| ThermalChipSimpleBoundaryOO N=40, M=40, P=40 | ~1M | ~64k | OOM after 5m 34.814s | 0.492s | N/A          | 2.696s           |
| ThermalChipSimpleBoundaryOO N=100, M=100, P=100 | ~16M | ~1M | N/A                  | 1.108s | N/A          | 28.640s           |

**Status**:
- Simulation parameters:
    - start time: `0`
    - end time: `1`
    - time-step: `0.001`

## ThermalChipCooling

| Name  | Vars | States | Compile Time OMC      | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|-----------------------|--------------------|--------------|---------------|
| ThermalChipCoolingSimpleBoundaryODE N=4, M=4, P=4 | 145 | 80 | 0.403s                | 1.035s             | 0.021s        | 0.011s        |
| ThermalChipCoolingSimpleBoundaryODE N=40, M=40, P=40 | ~145k | ~65k | 3m 24.333s            | 1.175s             | 1.785s       | 1.378s           |
| ThermalChipCoolingSimpleBoundaryODE N=100, M=100, P=100 | ~2M | ~1M | Compilation error (*) | 6.712s                | N/A          | 5.585s           |

**Status**:
- Simulation parameters:
    - start time: `0`
    - end time: `1`
    - time-step: `0.001`
- Implicit ranges in equations (e.g. `x[:,:] = ...`) have to be manually converted to explicit ones.

(*) Log from OMC:
```
Error: Internal error SimCodeUtil.simEqSystemIndex failed
[/var/lib/jenkins2/ws/LINUX_BUILDS/tmp.build/openmodelica-1.21.0~dev-206-gf7bc46b/OMCompiler/Compiler/SimCode/SerializeModelInfo.mo:116:9-116:84:writable] Error: Internal error SerializeModelInfo.serialize failed
[/var/lib/jenkins2/ws/LINUX_BUILDS/tmp.build/openmodelica-1.21.0~dev-206-gf7bc46b/OMCompiler/Compiler/SimCode/SimCodeMain.mo:530:7-530:78:writable] Error: Internal error /usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so: (null) failed
Error: Template error: A template call failed (/usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so: (null)). One possible reason could be that a template imported function call failed (which should not happen for functions called from within template code; templates assert pure 'match'/non-failing semantics).
[/var/lib/jenkins2/ws/LINUX_BUILDS/tmp.build/openmodelica-1.21.0~dev-206-gf7bc46b/OMCompiler/Compiler/SimCode/SimCodeMain.mo:530:7-530:78:writable] Error: Internal error /usr/bin/../lib/x86_64-linux-gnu/omc/libOpenModelicaCompiler.so: (null) failed
```

| Name  | Vars | States | Compile Time OMC     | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|----------------------| -------------------|--------------|---------------|
| ThermalChipCoolingSimpleBoundaryOO N=4, M=4, P=4 | ~1K | 80 | 1.494s               | 0.478s | 0.094s       | 0.013s        |
| ThermalChipCoolingSimpleBoundaryOO N=40, M=40, P=40 | ~500k | ~65k | OOM after 5m 36.858s | 0.651s | N/A          | 2.589s           |
| ThermalChipCoolingSimpleBoundaryOO N=100, M=100, P=100 | ~8M | ~1M | N/A                  | 1.545s | N/A          | 29.126s           |

**Status**:
- Simulation parameters:
    - start time: `0`
    - end time: `1`
    - time-step: `0.001`
- Implicit ranges in equations (e.g. `x[:,:] = ...`) have to be manually converted to explicit ones.

## MethanolHeatExchangers

| Name  | Vars | States | Compile Time OMC       | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------|------------------------|--------------------|--------------|---------------|
| MethanolHeatExchangers Nu=3, Nh=4, Nv=6 | 546 | 147 | 0.619s                 | 0.290s             | 0.056s       | 0.068s        |
| MethanolHeatExchangers Nu=30, Nh=40, Nv=20 | ~100k | ~24k | 4m 7.762s              | 0.495s             | 18.378s      | 11.919s       |
| MethanolHeatExchangers Nu=300, Nh=40, Nv=20 | ~1M | ~240k | OOM after 211m 27.277s | 1.835s             | N/A          | 1m 19.284s    |

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

**Status**:
 - "If equations" are not supported yet.

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| PowerGridSecondaryDAE Ne=2 | 158 | 17 | TBD | TBD | TBD | TBD |
| PowerGridSecondaryDAE Ne=100 | ~400k | ~40k | TBD | TBD | TBD | TBD |
| PowerGridSecondaryDAE Ne=400 | ~6M | ~640k | TBD | TBD | TBD | TBD |

**Status**:
 - "If equations" are not supported yet.

| Name  | Vars | States | Compile Time OMC  | Compile Time MARCO | Run Time OMC | RunTime Marco |
|  ---- |  ----| -------| -----------       | -------------------| ------------ | ------------- |
| PowerGridBaseOO Ne=2 | 364 | 16 | TBD | TBD | TBD | TBD |
| PowerGridBaseOO Ne=100 | ~800k | ~40k | TBD | TBD | TBD | TBD |
| PowerGridBaseOO Ne=400 | ~12M | ~600k | TBD | TBD | TBD | TBD |

**Status**:

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
