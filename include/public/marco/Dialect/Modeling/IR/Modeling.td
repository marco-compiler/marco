#ifndef MARCO_DIALECTS_MODELING_IR_MODELING_TD
#define MARCO_DIALECTS_MODELING_IR_MODELING_TD

include "marco/Dialect/Modeling/IR/ModelingAttributes.td"
include "marco/Dialect/Modeling/IR/ModelingTypes.td"
include "marco/Dialect/Modeling/IR/ModelingDialect.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//===---------------------------------------------------------------------===//
// Modeling operation definitions
//===---------------------------------------------------------------------===//

class Modeling_Op<string mnemonic, list<Trait> traits = []> :
    Op<Modeling_Dialect, mnemonic, traits>;

#endif // MARCO_DIALECTS_MODELING_IR_MODELING_TD
