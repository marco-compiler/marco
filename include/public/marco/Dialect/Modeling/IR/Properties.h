#ifndef MARCO_DIALAECT_MODELING_IR_PROPERTIES_H
#define MARCO_DIALAECT_MODELING_IR_PROPERTIES_H

#include "marco/Dialect/Modeling/IR/IndexSet.h"
#include "marco/Dialect/Modeling/IR/MultidimensionalRange.h"
#include "marco/Dialect/Modeling/IR/Point.h"
#include "marco/Dialect/Modeling/IR/Range.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"

namespace mlir::modeling
{
  //===-------------------------------------------------------------------===//
  // IndexSet
  //===-------------------------------------------------------------------===//

  mlir::LogicalResult setPropertiesFromAttribute(
      IndexSet& prop,
      mlir::Attribute attr,
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

  mlir::ArrayAttr getPropertiesAsAttribute(
      mlir::MLIRContext* context, const IndexSet& prop);

  llvm::hash_code computeHash(const IndexSet& prop);

  mlir::LogicalResult readFromMlirBytecode(
      mlir::DialectBytecodeReader& reader,
      IndexSet& prop);

  void writeToMlirBytecode(
      mlir::DialectBytecodeWriter& writer,
      const IndexSet& prop);

  mlir::LogicalResult parse(mlir::OpAsmParser& parser, IndexSet& prop);

  void print(mlir::OpAsmPrinter& printer, const IndexSet& prop);

  //===-------------------------------------------------------------------===//
  // IndexSetsList
  //===-------------------------------------------------------------------===//

  using IndexSetsList = llvm::SmallVector<IndexSet, 10>;

  mlir::LogicalResult setPropertiesFromAttribute(
      IndexSetsList& prop,
      mlir::Attribute attr,
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

  mlir::ArrayAttr getPropertiesAsAttribute(
      mlir::MLIRContext* context, const IndexSetsList& prop);

  llvm::hash_code computeHash(const IndexSetsList& prop);

  mlir::LogicalResult readFromMlirBytecode(
      mlir::DialectBytecodeReader& reader,
      IndexSetsList& prop);

  void writeToMlirBytecode(
      mlir::DialectBytecodeWriter& writer,
      const IndexSetsList& prop);

  mlir::LogicalResult parse(mlir::OpAsmParser& parser, IndexSetsList& prop);

  void print(mlir::OpAsmPrinter& printer, const IndexSetsList& prop);
}

#endif // MARCO_DIALAECT_MODELING_IR_PROPERTIES_H
