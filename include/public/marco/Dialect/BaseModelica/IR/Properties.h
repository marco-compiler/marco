#ifndef MARCO_DIALECT_BASEMODELICA_IR_PROPERTIES_H
#define MARCO_DIALECT_BASEMODELICA_IR_PROPERTIES_H

#include "marco/Dialect/BaseModelica/IR/DerivativesMap.h"
#include "marco/Dialect/BaseModelica/IR/VariableAccess.h"
#include "marco/Dialect/Modeling/IR/Properties.h"
#include "marco/Modeling/Scheduling.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::bmodelica {
using Point = mlir::modeling::Point;
using Range = mlir::modeling::Range;
using MultidimensionalRange = mlir::modeling::MultidimensionalRange;
using IndexSet = mlir::modeling::IndexSet;

//===---------------------------------------------------------------------===//
// Variable
//===---------------------------------------------------------------------===//

struct Variable {
  mlir::SymbolRefAttr name;
  IndexSet indices;

  Variable();

  Variable(mlir::SymbolRefAttr name, IndexSet indices);

  Variable(const IndexSet &equationIndices, const VariableAccess &access);

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
};

mlir::LogicalResult parse(mlir::OpAsmParser &parser, Variable &prop);

void print(mlir::OpAsmPrinter &printer, const Variable &prop);

//===---------------------------------------------------------------------===//
// VariablesList
//===---------------------------------------------------------------------===//

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

//===---------------------------------------------------------------------===//
// Schedule
//===---------------------------------------------------------------------===//

using Schedule = ::marco::modeling::scheduling::Direction;

mlir::LogicalResult setPropertiesFromAttribute(
    Schedule &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

mlir::Attribute getPropertiesAsAttribute(mlir::MLIRContext *context,
                                         const Schedule &prop);

llvm::hash_code computeHash(const Schedule &prop);

mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         Schedule &prop);

void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         const Schedule &prop);

mlir::LogicalResult parse(mlir::OpAsmParser &parser, Schedule &prop);

void print(mlir::OpAsmPrinter &printer, const Schedule &prop);

//===---------------------------------------------------------------------===//
// ScheduleList
//===---------------------------------------------------------------------===//

using ScheduleList = llvm::SmallVector<Schedule, 10>;

mlir::LogicalResult setPropertiesFromAttribute(
    ScheduleList &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

mlir::ArrayAttr getPropertiesAsAttribute(mlir::MLIRContext *context,
                                         const ScheduleList &prop);

llvm::hash_code computeHash(const ScheduleList &prop);

mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         ScheduleList &prop);

void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         const ScheduleList &prop);

mlir::LogicalResult parse(mlir::OpAsmParser &parser, ScheduleList &prop);

void print(mlir::OpAsmPrinter &printer, const ScheduleList &prop);
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_IR_PROPERTIES_H
