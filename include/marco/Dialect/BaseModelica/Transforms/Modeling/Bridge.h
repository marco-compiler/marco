#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_BRIDGE_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_BRIDGE_H

#include "marco/Dialect/BaseModelica/Transforms/Modeling/EquationBridge.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/SCCBridge.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/VariableBridge.h"

namespace mlir::bmodelica::bridge {
class Storage {
public:
  llvm::SmallVector<std::unique_ptr<VariableBridge>> variableBridges;
  VariablesMap variablesMap;
  llvm::SmallVector<std::unique_ptr<EquationBridge>> equationBridges;
  EquationsMap equationsMap;
  llvm::SmallVector<std::unique_ptr<SCCBridge>> sccBridges;

protected:
  // Enforce heap-allocation to ensure that equation bridges do not end up with
  // dangling pointers to the storage class.
  Storage() = default;

public:
  Storage(const Storage &other) = delete;
  Storage(Storage &&other) = default;
  virtual ~Storage() = default;
  Storage &operator=(const Storage &other) = delete;
  Storage &operator=(Storage &&other) = default;

  static std::unique_ptr<Storage> create();

  VariableBridge &addVariable(mlir::SymbolRefAttr name, IndexSet indices);
  VariableBridge &addVariable(VariableOp variableOp);

  EquationBridge &
  addEquation(uint64_t id, EquationInstanceOp op,
              mlir::SymbolTableCollection &symbolTableCollection);

  void clear();

  void clearVariables();

  void clearEquations();

  void clearSCCs();
};
} // namespace mlir::bmodelica::bridge

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_BRIDGE_H
