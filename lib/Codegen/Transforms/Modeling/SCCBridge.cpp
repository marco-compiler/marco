#include "marco/Codegen/Transforms/Modeling/SCCBridge.h"

using namespace ::mlir::modelica::bridge;

namespace mlir::modelica::bridge
{
  SCCBridge::SCCBridge(
      SCCOp op,
      mlir::SymbolTableCollection& symbolTable,
      WritesMap<VariableOp, MatchedEquationInstanceOp>& writesMap,
      llvm::DenseMap<
          MatchedEquationInstanceOp, MatchedEquationBridge*>& equationsMap)
      : op(op),
        symbolTable(&symbolTable),
        writesMap(&writesMap),
        equationsMap(&equationsMap)
  {
  }
}
