#ifndef MARCO_DIALECT_RUNTIME_IR_PROPERTIES_H
#define MARCO_DIALECT_RUNTIME_IR_PROPERTIES_H

#include "marco/Dialect/Modeling/IR/Properties.h"
#include <variant>

namespace mlir::runtime {
using Point = ::mlir::modeling::Point;
using Range = ::mlir::modeling::Range;
using MultidimensionalRange = ::mlir::modeling::MultidimensionalRange;
using IndexSet = ::mlir::modeling::IndexSet;

//===-------------------------------------------------------------------===//
// PrintableIndicesList
//===-------------------------------------------------------------------===//

class PrintInfo {
public:
  explicit PrintInfo(bool value);

  explicit PrintInfo(IndexSet value);

  bool operator==(const PrintInfo &other) const;

  template <typename T>
  [[nodiscard]] bool isa() const {
    return std::holds_alternative<T>(data);
  }

  template <typename T>
  const T &get() const {
    return std::get<T>(data);
  }

  friend llvm::hash_code hash_value(const PrintInfo &printInfo);

private:
  std::variant<bool, IndexSet> data;
};

using PrintableIndicesList = llvm::SmallVector<PrintInfo>;

mlir::LogicalResult setPropertiesFromAttribute(
    PrintableIndicesList &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError);

mlir::ArrayAttr getPropertiesAsAttribute(mlir::MLIRContext *context,
                                         const PrintableIndicesList &prop);

llvm::hash_code computeHash(const PrintableIndicesList &prop);

mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         PrintableIndicesList &prop);

void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         const PrintableIndicesList &prop);
} // namespace mlir::runtime

#endif // MARCO_DIALECT_RUNTIME_IR_PROPERTIES_H
