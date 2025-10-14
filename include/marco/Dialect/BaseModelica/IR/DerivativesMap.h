#ifndef MARCO_DIALECT_BASEMODELICA_IR_DERIVATIVESMAP_H
#define MARCO_DIALECT_BASEMODELICA_IR_DERIVATIVESMAP_H

#include "marco/Modeling/IndexSet.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/MapVector.h"
#include <mutex>

namespace mlir::bmodelica {
class DerivativesMap {
public:
  virtual ~DerivativesMap() = default;

  bool operator==(const DerivativesMap &other) const;

  mlir::DictionaryAttr asAttribute(mlir::MLIRContext *context) const;

  static mlir::LogicalResult
  setFromAttr(DerivativesMap &prop, mlir::Attribute attr,
              llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

  [[nodiscard]] llvm::hash_code hash() const;

  friend mlir::LogicalResult
  readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                       DerivativesMap &prop);

  friend void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                                  DerivativesMap &prop);

  friend mlir::LogicalResult parse(mlir::OpAsmParser &parser,
                                   DerivativesMap &prop);

  friend void print(mlir::OpAsmPrinter &printer, const DerivativesMap &prop);

  virtual bool empty() const;

  virtual llvm::SmallVector<mlir::SymbolRefAttr> getDerivedVariables() const;

  /// Get the derivative variable of a given state variable.
  virtual std::optional<mlir::SymbolRefAttr>
  getDerivative(mlir::SymbolRefAttr variable) const;

  /// Set the derivative variable for a state one.
  virtual void setDerivative(mlir::SymbolRefAttr variable,
                             mlir::SymbolRefAttr derivative);

  virtual std::optional<std::reference_wrapper<const marco::modeling::IndexSet>>
  getDerivedIndices(mlir::SymbolRefAttr variable) const;

  virtual void setDerivedIndices(mlir::SymbolRefAttr variable,
                                 marco::modeling::IndexSet indices);

  virtual void addDerivedIndices(mlir::SymbolRefAttr variable,
                                 marco::modeling::IndexSet indices);

  /// Get the state variable of a given derivative variable.
  virtual std::optional<mlir::SymbolRefAttr>
  getDerivedVariable(mlir::SymbolRefAttr derivative) const;

private:
  llvm::MapVector<mlir::SymbolRefAttr, mlir::SymbolRefAttr> derivatives;

  llvm::MapVector<mlir::SymbolRefAttr, marco::modeling::IndexSet>
      derivedIndices;

  llvm::MapVector<mlir::SymbolRefAttr, mlir::SymbolRefAttr> inverseDerivatives;
};

/// A multithread safe version of DerivativesMap.
class LockedDerivativesMap : public DerivativesMap {
  DerivativesMap &derivativesMap;
  mutable std::mutex mutex;

public:
  explicit LockedDerivativesMap(DerivativesMap &derivativesMap);

  LockedDerivativesMap(const LockedDerivativesMap &other) = delete;
  LockedDerivativesMap(LockedDerivativesMap &&other) = delete;
  LockedDerivativesMap &operator=(const LockedDerivativesMap &other) = delete;
  LockedDerivativesMap &operator=(LockedDerivativesMap &&other) = delete;

  bool empty() const override;

  llvm::SmallVector<mlir::SymbolRefAttr> getDerivedVariables() const override;

  std::optional<mlir::SymbolRefAttr>
  getDerivative(mlir::SymbolRefAttr variable) const override;

  void setDerivative(mlir::SymbolRefAttr variable,
                     mlir::SymbolRefAttr derivative) override;

  std::optional<std::reference_wrapper<const marco::modeling::IndexSet>>
  getDerivedIndices(mlir::SymbolRefAttr variable) const override;

  void setDerivedIndices(mlir::SymbolRefAttr variable,
                         marco::modeling::IndexSet indices) override;

  void addDerivedIndices(mlir::SymbolRefAttr variable,
                         marco::modeling::IndexSet indices) override;

  std::optional<mlir::SymbolRefAttr>
  getDerivedVariable(mlir::SymbolRefAttr derivative) const override;
};
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_IR_DERIVATIVESMAP_H
