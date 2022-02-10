#ifndef MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD
#define MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD

include "marco/dialects/modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"

class Modelica_Attr<string name> : AttrDef<Modelica_Dialect, name> { }

def BooleanAttr : Modelica_Attr<"Boolean"> {
  let mnemonic = "bool";

  let summary = "Modelica's Boolean attribute";

  let description = [{
    Type representing Modelica's Boolean.
  }];

  let printer = [{
    $_printer << "bool";
  }];

  let parser = [{
    return get($_ctxt);
  }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD
