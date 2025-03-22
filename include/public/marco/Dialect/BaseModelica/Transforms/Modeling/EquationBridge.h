#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_EQUATIONBRIDGE_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_EQUATIONBRIDGE_H

#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/Modeling/VariableBridge.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
class raw_ostream;
}

namespace mlir::bmodelica::bridge {
class EquationBridge {
public:
  using VariablesMap = llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *>;

private:
  int64_t id;
  EquationInstanceOp op;
  mlir::SymbolTableCollection *symbolTable;
  VariableAccessAnalysis *accessAnalysis;
  VariablesMap *variablesMap;

public:
  template <typename... Args>
  static std::unique_ptr<EquationBridge> build(Args &&...args) {
    return std::make_unique<EquationBridge>(std::forward<Args>(args)...);
  }

  EquationBridge(
      int64_t id, EquationInstanceOp op,
      mlir::SymbolTableCollection &symbolTable,
      llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> &variablesMap);

  EquationBridge(
      int64_t id, EquationInstanceOp op,
      mlir::SymbolTableCollection &symbolTable,
      VariableAccessAnalysis &accessAnalysis,
      llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge *> &variablesMap);

  // Forbid copies to avoid dangling pointers by design.
  EquationBridge(const EquationBridge &other) = delete;
  EquationBridge(EquationBridge &&other) = delete;
  EquationBridge &operator=(const EquationBridge &other) = delete;
  EquationBridge &operator==(const EquationBridge &other) = delete;

  friend llvm::hash_code hash_value(const EquationBridge &val);

  /// @Getters
  /// {

  int64_t getId() const;

  EquationInstanceOp getOp() const;

  mlir::SymbolTableCollection &getSymbolTableCollection();

  const mlir::SymbolTableCollection &getSymbolTableCollection() const;

  bool hasAccessAnalysis() const;

  VariableAccessAnalysis &getAccessAnalysis();

  const VariableAccessAnalysis &getAccessAnalysis() const;

  VariablesMap &getVariablesMap();

  const VariablesMap &getVariablesMap() const;

  /// }
};

llvm::hash_code hash_value(const EquationBridge *val);
} // namespace mlir::bmodelica::bridge

namespace marco::modeling::matching {
template <>
struct EquationTraits<::mlir::bmodelica::bridge::EquationBridge *> {
  using Equation = ::mlir::bmodelica::bridge::EquationBridge *;
  using Id = int64_t;

  static Id getId(const Equation *equation);

  static size_t getNumOfIterationVars(const Equation *equation);

  static IndexSet getIterationRanges(const Equation *equation);

  using VariableType = ::mlir::bmodelica::bridge::VariableBridge *;
  using AccessProperty = ::mlir::bmodelica::EquationPath;

  static std::vector<Access<VariableType, AccessProperty>>
  getAccesses(const Equation *equation);

  static llvm::raw_ostream &dump(const Equation *equation,
                                 llvm::raw_ostream &os);

private:
  static std::vector<Access<VariableType, AccessProperty>>
  convertAccesses(const Equation *equation,
                  llvm::ArrayRef<mlir::bmodelica::VariableAccess> accesses);
};
} // namespace marco::modeling::matching

namespace marco::modeling::dependency {
template <>
struct EquationTraits<::mlir::bmodelica::bridge::EquationBridge *> {
  using Equation = ::mlir::bmodelica::bridge::EquationBridge *;
  using Id = int64_t;

  static Id getId(const Equation *equation);

  static size_t getNumOfIterationVars(const Equation *equation);

  static IndexSet getIterationRanges(const Equation *equation);

  using VariableType = ::mlir::bmodelica::bridge::VariableBridge *;
  using VariableAccess = mlir::bmodelica::VariableAccess;
  using AccessProperty = VariableAccess;

  static std::vector<Access<VariableType, AccessProperty>>
  getAccesses(const Equation *equation);

  static std::vector<Access<VariableType, AccessProperty>>
  getWrites(const Equation *equation);

  static std::vector<Access<VariableType, AccessProperty>>
  getReads(const Equation *equation);

  static llvm::raw_ostream &dump(const Equation *equation,
                                 llvm::raw_ostream &os);

private:
  static std::vector<Access<VariableType, AccessProperty>>
  convertAccesses(const Equation *equation,
                  llvm::ArrayRef<mlir::bmodelica::VariableAccess> accesses);
};
} // namespace marco::modeling::dependency

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_EQUATIONBRIDGE_H
