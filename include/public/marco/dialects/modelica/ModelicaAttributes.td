#ifndef MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD
#define MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD

include "marco/dialects/modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Modelica attribute definitions
//===----------------------------------------------------------------------===//

class Modelica_Attr<string name, string attrMnemonic>
    : AttrDef<Modelica_Dialect, name> {
  let mnemonic = attrMnemonic;
}

def BooleanAttr : Modelica_Attr<"Boolean", "bool"> {
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
