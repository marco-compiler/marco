#ifndef MARCO_DIALECTS_MODELICA_MODELICATYPES_TD
#define MARCO_DIALECTS_MODELICA_MODELICATYPES_TD

include "marco/dialects/modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"

class Modelica_Type<string name> : TypeDef<Modelica_Dialect, name> { }

def BooleanType : Modelica_Type<"Boolean"> {
  let mnemonic = "bool";

  let summary = "Modelica's Boolean type";

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

def IntegerType : Modelica_Type<"Integer"> {
  let mnemonic = "int";

  let summary = "Modelica's Integer type";

  let description = [{
    Type representing Modelica's Integer.
  }];

  let printer = [{
    $_printer << "int";
  }];

  let parser = [{
    return get($_ctxt);
  }];
}

def RealType : Modelica_Type<"Real"> {
  let mnemonic = "real";

  let summary = "Modelica's Real type";

  let description = [{
    Type representing Modelica's Real.
  }];

  let printer = [{
    $_printer << "real";
  }];

  let parser = [{
    return get($_ctxt);
  }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICATYPES_TD
