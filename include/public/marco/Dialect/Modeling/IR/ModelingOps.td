#ifndef MARCO_DIALECTS_MODELING_IR_MODELINGOPS_TD
#define MARCO_DIALECTS_MODELING_IR_MODELINGOPS_TD

include "marco/Dialect/Modeling/IR/ModelingAttributes.td"
include "marco/Dialect/Modeling/IR/Modeling.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//===---------------------------------------------------------------------===//
// Modeling operation definitions
//===---------------------------------------------------------------------===//

class Modeling_Op<string mnemonic, list<Trait> traits = []> :
    Op<Modeling_Dialect, mnemonic, traits>;

#endif // MARCO_DIALECTS_MODELING_IR_MODELINGOPS_TD
