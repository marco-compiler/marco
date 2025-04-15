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
class Storage;

class EquationBridge {
public:
  using Id = uint64_t;

  /// Wrapper for the list of accesses to variables.
  /// The list of accesses can be referenced or owned by the class, thus
  /// allowing to hide the caching level to downstream users.
  class AccessesList {
    using Reference = llvm::ArrayRef<VariableAccess>;
    using Container = llvm::SmallVector<VariableAccess>;

    std::variant<Reference, Container> accesses;

  public:
    explicit AccessesList(Reference accesses);
    explicit AccessesList(Container accesses);

    operator llvm::ArrayRef<VariableAccess>() const;

    auto begin() const {
      return static_cast<llvm::ArrayRef<VariableAccess>>(*this).begin();
    }

    auto end() const {
      return static_cast<llvm::ArrayRef<VariableAccess>>(*this).end();
    }
  };

private:
  /// Unique identifier.
  const Id id;

  /// IR operation.
  EquationInstanceOp op;

  const Storage &storage;
  mlir::SymbolTableCollection *symbolTable;
  VariableAccessAnalysis *accessAnalysis;

protected:
  // The constructors are not made public in order to enforce the construction
  // through the Storage class.
  EquationBridge(uint64_t id, EquationInstanceOp op,
                 mlir::SymbolTableCollection &symbolTableCollection,
                 const Storage &storage);

public:
  // Forbid copies to avoid dangling pointers by design.
  EquationBridge(const EquationBridge &other) = delete;
  EquationBridge(EquationBridge &&other) = delete;
  EquationBridge &operator=(const EquationBridge &other) = delete;
  EquationBridge &operator==(const EquationBridge &other) = delete;

  friend llvm::hash_code hash_value(const EquationBridge &val);

  /// @Getters
  /// {

  Id getId() const;

  EquationInstanceOp getOp() const;

  mlir::SymbolTableCollection &getSymbolTableCollection();

  const mlir::SymbolTableCollection &getSymbolTableCollection() const;

  bool hasAccessAnalysis() const;

  VariableAccessAnalysis &getAccessAnalysis();

  const VariableAccessAnalysis &getAccessAnalysis() const;

  const VariablesMap &getVariablesMap() const;

  /// }
  /// @name Setters.
  /// {

  void setAccessAnalysis(VariableAccessAnalysis &accessAnalysis);

  /// }

  size_t getOriginalRank() const;

  IndexSet getOriginalIndices() const;

  AccessesList getOriginalAccesses();

  AccessesList getOriginalWriteAccesses();

  AccessesList getOriginalReadAccesses();

  size_t getRank() const;

  IndexSet getIndices() const;

  using AccessWalkFn = llvm::function_ref<void(
      const VariableAccess &access, VariableBridge *variable,
      const AccessFunction &accessFunction)>;

  void walkAccesses(AccessWalkFn callbackFn);

  void walkWriteAccesses(AccessWalkFn callbackFn);

  void walkReadAccesses(AccessWalkFn callbackFn);
};

llvm::hash_code hash_value(const EquationBridge *val);

using EquationsMap = llvm::DenseMap<EquationBridge::Id, EquationBridge *>;
} // namespace mlir::bmodelica::bridge

namespace marco::modeling::matching {
template <>
struct EquationTraits<::mlir::bmodelica::bridge::EquationBridge *> {
  using Equation = ::mlir::bmodelica::bridge::EquationBridge *;
  using Id = ::mlir::bmodelica::bridge::EquationBridge::Id;

  static Id getId(const Equation *equation);

  static size_t getNumOfIterationVars(const Equation *equation);

  static IndexSet getIndices(const Equation *equation);

  using VariableType = ::mlir::bmodelica::bridge::VariableBridge *;
  using VariableId = VariableTraits<VariableType>::Id;

  static std::vector<Access<VariableId>> getAccesses(const Equation *equation);

  static llvm::raw_ostream &dump(const Equation *equation,
                                 llvm::raw_ostream &os);

private:
  static Access<VariableId>
  convertAccess(const mlir::bmodelica::VariableAccess &access,
                mlir::bmodelica::bridge::VariableBridge *variable,
                const AccessFunction &accessFunction);
};
} // namespace marco::modeling::matching

namespace marco::modeling::dependency {
template <>
struct EquationTraits<::mlir::bmodelica::bridge::EquationBridge *> {
  using Equation = ::mlir::bmodelica::bridge::EquationBridge *;
  using Id = ::mlir::bmodelica::bridge::EquationBridge::Id;

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
  static Access<VariableType, AccessProperty>
  convertAccess(const mlir::bmodelica::VariableAccess &access,
                mlir::bmodelica::bridge::VariableBridge *variable,
                const AccessFunction &accessFunction);
};
} // namespace marco::modeling::dependency

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_EQUATIONBRIDGE_H
