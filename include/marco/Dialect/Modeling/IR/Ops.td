#ifndef MARCO_DIALECTS_MODELING_IR_OPS_TD
#define MARCO_DIALECTS_MODELING_IR_OPS_TD

include "marco/Dialect/Modeling/IR/Attributes.td"
include "marco/Dialect/Modeling/IR/Dialect.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//===---------------------------------------------------------------------===//
// Modeling operation definitions
//===---------------------------------------------------------------------===//

class Modeling_Op<string mnemonic, list<Trait> traits = []> :
    Op<Modeling_Dialect, mnemonic, traits>;

#endif // MARCO_DIALECTS_MODELING_IR_OPS_TD
