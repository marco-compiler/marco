#include "marco/Codegen/Transforms/Modeling/EquationBridge.h"

using namespace ::mlir::modelica::bridge;

namespace mlir::modelica::bridge
{
  EquationBridge::EquationBridge(
      EquationInstanceOp op,
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
