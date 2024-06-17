OMC settings for MARCO input file
=================================

MARCO starts from an array-preserving flat Modelica output, which is obtained using the OMC front end.
The format is inspired as much as possible from the upcoming Flat Modelica standard, see
[MCP 0031](https://github.com/modelica/ModelicaSpecification/tree/MCP/0031/RationaleMCP/0031), which is
however currently limited to fully scalarized flat output.

Some extensions are then introduced to handle non-scalarized arrays.

By default, the OMC front end does not evaluate the parameters, so they can be handled by the generated code
and changed at runtime without the need of recompiling the model. However, this typically introduces some overhead
and prevents some code optimization. If the aim is to run the models as fast as possible, the reference performance
is the one obtained by evaluating all parameters at compile time. This is easily achieved by setting the
`-d=evaluateAllParameters` compiler flag in OMC, which causes the front end to automatically evaluate all parameters,
including parameters depending on other parameters, generating a flat code that only contains literal parameter values.

Later on this could be improved by leaving out some parameters that could be changed at runtime, in case one wants to
perform, e.g., sensitivity or parameter optimization studies.

Full set of compiler flags
--------------------------
Array based models:
```
--baseModelica --newBackend -d=evaluateAllParameters
```
Automatic vectorization of model with many scalar instances of the same components:
```
--baseModelica --newBackend -d=mergeComponents,evaluateAllParameters
```
If one wants the OMC frontend to also do function inlining and flatten record definitions and record equations, add the `--frontendInline` flag.

These flags should be set in [FrontendActions.cpp](https://github.com/modelica-polimi/marco/blob/master/lib/Frontend/FrontendActions.cpp), around line 248.

Explanations
-----------
- [``--baseModelica``](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/omchelptext.html#omcflag-basemodelica)
  Enables Flat Modelica Output (see [MCP 0031](https://github.com/modelica/ModelicaSpecification/tree/MCP/0031/RationaleMCP/0031)).
- [``--newBackend``](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/omchelptext.html#omcflag-newbackend)
  Disables scalarization pass in OMC front end.
- [``-d=mergeComponents``](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/omchelptext.html#omcflag-debug-mergecomponents)
  Merges instances of the same model with the same type of modifications into arrays.
- [``-d=evaluateAllParameters``](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/omchelptext.html#omcflag-debug-evaluateallparameters)
  Evaluate all parameters in the frontend and produce literal values only for parameter modifiers.
