#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_VARIABLEBRIDGE_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_MODELING_VARIABLEBRIDGE_H

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Modeling/Dependency.h"
#include "marco/Modeling/IndexSet.h"
#include "marco/Modeling/Matching.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace llvm {
class raw_ostream;
}

namespace mlir::bmodelica::bridge {
class VariableBridge {
public:
  struct Id {
    mlir::SymbolRefAttr name;

    Id(mlir::SymbolRefAttr name);

    bool operator<(const Id &other) const;

    bool operator==(const Id &other) const;

    bool operator!=(const Id &other) const;
  };

  Id id;
  mlir::SymbolRefAttr name;
  marco::modeling::IndexSet indices;

public:
  static std::unique_ptr<VariableBridge> build(mlir::SymbolRefAttr name,
                                               IndexSet indices);

  static std::unique_ptr<VariableBridge> build(VariableOp variable);

  VariableBridge(mlir::SymbolRefAttr name, marco::modeling::IndexSet indices);

  // Forbid copies to avoid dangling pointers by design.
  VariableBridge(const VariableBridge &other) = delete;
  VariableBridge(VariableBridge &&other) = delete;
  VariableBridge &operator=(const VariableBridge &other) = delete;
  VariableBridge &operator==(const VariableBridge &other) = delete;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const VariableBridge::Id &obj);
} // namespace mlir::bmodelica::bridge

namespace llvm {
template <>
struct DenseMapInfo<::mlir::bmodelica::bridge::VariableBridge::Id> {
  static inline ::mlir::bmodelica::bridge::VariableBridge::Id getEmptyKey() {
    return {nullptr};
  }

  static inline ::mlir::bmodelica::bridge::VariableBridge::Id
  getTombstoneKey() {
    return {nullptr};
  }

  static unsigned
  getHashValue(const ::mlir::bmodelica::bridge::VariableBridge::Id &val) {
    return llvm::DenseMapInfo<mlir::SymbolRefAttr>::getHashValue(val.name);
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
