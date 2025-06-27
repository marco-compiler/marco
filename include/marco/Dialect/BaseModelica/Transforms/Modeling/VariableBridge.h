#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_VARIABLEBRIDGE_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_VARIABLEBRIDGE_H

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Modeling/Dependency.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/Matching.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
class raw_ostream;
}

namespace mlir::bmodelica::bridge {
class Storage;

class VariableBridge {
public:
  class Id {
    mlir::SymbolRefAttr name;

  public:
    Id(mlir::SymbolRefAttr name);

    bool operator==(const Id &other) const;

    bool operator!=(const Id &other) const;

    bool operator<(const Id &other) const;

    friend llvm::hash_code hash_value(const Id &val);

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Id &obj);
  };

private:
  /// Unique identifier.
  const Id id;

  /// The name of the variable.
  const mlir::SymbolRefAttr name;

  /// The indices of the variable.
  const IndexSet indices;

protected:
  // The constructor is not made public in order to enforce the construction
  // through the Storage class.
  VariableBridge(mlir::SymbolRefAttr name, IndexSet indices);

public:
  // Forbid copies to avoid dangling pointers by design.
  VariableBridge(const VariableBridge &other) = delete;
  VariableBridge(VariableBridge &&other) = delete;
  VariableBridge &operator=(const VariableBridge &other) = delete;
  VariableBridge &operator==(const VariableBridge &other) = delete;

  friend llvm::hash_code hash_value(const VariableBridge &val);

  /// @name Getters.
  /// {

  const Id &getId() const { return id; }

  mlir::SymbolRefAttr getName() const { return name; }

  const IndexSet &getOriginalIndices() const;

  IndexSet getIndices() const;

  /// }
};

using VariablesMap = llvm::DenseMap<VariableBridge::Id, VariableBridge *>;
} // namespace mlir::bmodelica::bridge

namespace llvm {
template <>
struct DenseMapInfo<::mlir::bmodelica::bridge::VariableBridge::Id> {
  static ::mlir::bmodelica::bridge::VariableBridge::Id getEmptyKey() {
    return {llvm::DenseMapInfo<mlir::SymbolRefAttr>::getEmptyKey()};
  }

  static ::mlir::bmodelica::bridge::VariableBridge::Id getTombstoneKey() {
    return {llvm::DenseMapInfo<mlir::SymbolRefAttr>::getTombstoneKey()};
  }

  static unsigned
  getHashValue(const ::mlir::bmodelica::bridge::VariableBridge::Id &val) {
    return hash_value(val);
  }

  static bool
  isEqual(const ::mlir::bmodelica::bridge::VariableBridge::Id &lhs,
          const ::mlir::bmodelica::bridge::VariableBridge::Id &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace marco::modeling::matching {
template <>
struct VariableTraits<::mlir::bmodelica::bridge::VariableBridge *> {
  using Variable = ::mlir::bmodelica::bridge::VariableBridge *;
  using Id = ::mlir::bmodelica::bridge::VariableBridge::Id;

  static Id getId(const Variable *variable);

  static size_t getRank(const Variable *variable);

  static IndexSet getIndices(const Variable *variable);

  static llvm::raw_ostream &dump(const Variable *variable,
                                 llvm::raw_ostream &os);
};
} // namespace marco::modeling::matching

namespace marco::modeling::dependency {
template <>
struct VariableTraits<::mlir::bmodelica::bridge::VariableBridge *> {
  using Variable = ::mlir::bmodelica::bridge::VariableBridge *;
  using Id = ::mlir::bmodelica::bridge::VariableBridge::Id;

  static Id getId(const Variable *variable);

  static size_t getRank(const Variable *variable);

  static IndexSet getIndices(const Variable *variable);

  static llvm::raw_ostream &dump(const Variable *variable,
                                 llvm::raw_ostream &os);
};
} // namespace marco::modeling::dependency

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_VARIABLEBRIDGE_H
