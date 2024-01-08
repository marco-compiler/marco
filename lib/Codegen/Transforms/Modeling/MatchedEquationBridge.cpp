#include "marco/Codegen/Transforms/Modeling/MatchedEquationBridge.h"

using namespace ::mlir::modelica::bridge;

namespace mlir::modelica::bridge
{
  MatchedEquationBridge::MatchedEquationBridge(
      MatchedEquationInstanceOp op,
      mlir::SymbolTableCollection& symbolTable,
      VariableAccessAnalysis& accessAnalysis,
      llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>& variablesMap)
      : op(op),
        symbolTable(&symbolTable),
        accessAnalysis(&accessAnalysis),
        variablesMap(&variablesMap)
  {
  }
}
