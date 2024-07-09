#ifndef MARCO_DIALECT_BASEMODELICA_IR_DERIVATIVESMAP_H
#define MARCO_DIALECT_BASEMODELICA_IR_DERIVATIVESMAP_H

#include "marco/Modeling/IndexSet.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::bmodelica
{
  class DerivativesMap
  {
    public:
      bool operator==(const DerivativesMap& other) const;

      mlir::DictionaryAttr asAttribute(mlir::MLIRContext* context) const;

      static mlir::LogicalResult setFromAttr(
          DerivativesMap& prop,
          mlir::Attribute attr,
          llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

      [[nodiscard]] llvm::hash_code hash() const;

      friend mlir::LogicalResult readFromMlirBytecode(
          mlir::DialectBytecodeReader& reader,
          DerivativesMap& prop);

      friend void writeToMlirBytecode(
          mlir::DialectBytecodeWriter& writer,
          DerivativesMap& prop);

      friend mlir::LogicalResult parse(
          mlir::OpAsmParser& parser, DerivativesMap& prop);

      friend void print(
          mlir::OpAsmPrinter& printer, const DerivativesMap& prop);

      bool empty() const;

      llvm::DenseSet<mlir::SymbolRefAttr> getDerivedVariables() const;

      /// Get the derivative variable of a given state variable.
      std::optional<mlir::SymbolRefAttr> getDerivative(
          mlir::SymbolRefAttr variable) const;

      /// Set the derivative variable for a state one.
      void setDerivative(
          mlir::SymbolRefAttr variable, mlir::SymbolRefAttr derivative);

      std::optional<std::reference_wrapper<const marco::modeling::IndexSet>>
      getDerivedIndices(mlir::SymbolRefAttr variable) const;

      void setDerivedIndices(
          mlir::SymbolRefAttr variable, marco::modeling::IndexSet indices);

      void addDerivedIndices(
          mlir::SymbolRefAttr variable, marco::modeling::IndexSet indices);

      /// Get the state variable of a given derivative variable.
      std::optional<mlir::SymbolRefAttr>
      getDerivedVariable(mlir::SymbolRefAttr derivative) const;

    private:
      llvm::DenseMap<
          mlir::SymbolRefAttr,
          mlir::SymbolRefAttr> derivatives;

      llvm::DenseMap<
          mlir::SymbolRefAttr,
          marco::modeling::IndexSet> derivedIndices;

      llvm::DenseMap<
          mlir::SymbolRefAttr,
          mlir::SymbolRefAttr> inverseDerivatives;
  };
}

#endif // MARCO_DIALECT_BASEMODELICA_IR_DERIVATIVESMAP_H
