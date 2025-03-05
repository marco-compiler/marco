#ifndef MARCO_DIALECT_BASEMODELICA_IR_PROPERTIES_H
#define MARCO_DIALECT_BASEMODELICA_IR_PROPERTIES_H

#include "marco/Dialect/BaseModelica/IR/DerivativesMap.h"
#include "marco/Dialect/Modeling/IR/Properties.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::bmodelica {
using Point = mlir::modeling::Point;
using Range = mlir::modeling::Range;
using MultidimensionalRange = mlir::modeling::MultidimensionalRange;
using IndexSet = mlir::modeling::IndexSet;

//===-------------------------------------------------------------------===//
// Variable
//===-------------------------------------------------------------------===//

struct Variable {
  mlir::SymbolRefAttr name;
  IndexSet indices;

  Variable();

  Variable(mlir::SymbolRefAttr name, IndexSet indices);

  bool operator==(const Variable &other) const;

  operator bool() const;

  mlir::Attribute asAttribute(mlir::MLIRContext *context) const;

  static mlir::LogicalResult
  setFromAttr(Variable &prop, mlir::Attribute attr,
              llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

  [[nodiscard]] llvm::hash_code hash() const;

  friend llvm::hash_code hash_value(const Variable &value);

  static mlir::LogicalResult
  readFromMlirBytecode(mlir::DialectBytecodeReader &reader, Variable &prop);

  void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer) const;

  friend mlir::LogicalResult parse(mlir::OpAsmParser &parser, Variable &prop);

  friend void print(mlir::OpAsmPrinter &printer, const Variable &prop);
};

//===-------------------------------------------------------------------===//
// VariablesList
//===-------------------------------------------------------------------===//

using VariablesList = llvm::SmallVector<Variable, 10>;

mlir::LogicalResult setPropertiesFromAttribute(
    VariablesList &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

mlir::ArrayAttr getPropertiesAsAttribute(mlir::MLIRContext *context,
                                         const VariablesList &prop);

llvm::hash_code computeHash(const VariablesList &prop);

mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         VariablesList &prop);

void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         VariablesList &prop);

mlir::LogicalResult parse(mlir::OpAsmParser &parser, VariablesList &prop);

void print(mlir::OpAsmPrinter &printer, const VariablesList &prop);
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_IR_PROPERTIES_H
