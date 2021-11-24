OMC settings for MARCO input file
=================================

MARCO starts from an array-preserving flat Modelica output, which is obtained using the OMC front end.
The format is inspired as much as possible from the upcoming Flat Modelica standard, see
[MCP 0031](https://github.com/modelica/ModelicaSpecification/tree/MCP/0031/RationaleMCP/0031), which is
however currently limited to fully scalarized flat output.

Some extensions are then introduced to handle non-scalarized arrays.

Full set of compiler flags
--------------------------
```
-f -d=nonfScalarize,arrayConnect,combineSubscripts,printRecordTypes --newBackend --showStructuralAnnotations
```

Explanations
-----------
- [``-f``](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/omchelptext.html#omcflag-flatmodelica)
  Enables Flat Modelica Output (see [MCP 0031](https://github.com/modelica/ModelicaSpecification/tree/MCP/0031/RationaleMCP/0031)).
- [``-d=nonfScalarize``](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/omchelptext.html#omcflag-debug-nfscalarize)
  Disables scalarization pass in OMC front end.
- [``-d=arrayConnect``](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/omchelptext.html#omcflag-debug-arrayconnect)
  Handles array connect equations.
- [``-d=combineSubscripts``](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/omchelptext.html#omcflag-debug-arrayconnect)
  Turns ``a[j].b[k]`` into ``a.b[j,k]``.
- [``-d=printRecordTypes``](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/omchelptext.html#omcflag-debug-printrecordtypes)
  Prints flat record type definitions instead of constructor function definitions, which is the default.
- [``--newBackend``](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/omchelptext.html#omcflag-newbackend)
  avoids flattening records in variable declarations.
- [``--showStructuralAnnotations``](https://openmodelica.org/doc/OpenModelicaUsersGuide/latest/omchelptext.html#omcflag-showstructuralannotations)
  keeps structural annotations in the function definitions, such as ``Inline = true``.

