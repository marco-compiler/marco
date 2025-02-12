#ifndef MARCO_DIALECT_MODELING_IR_ATTRIBUTES_H
#define MARCO_DIALECT_MODELING_IR_ATTRIBUTES_H

#include "marco/Dialect/Modeling/IR/MultidimensionalRange.h"
#include "marco/Dialect/Modeling/IR/Point.h"
#include "marco/Dialect/Modeling/IR/Range.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Modeling/IR/ModelingAttributes.h.inc"

namespace mlir {
template <>
struct FieldParser<mlir::modeling::Range> {
  static FailureOr<mlir::modeling::Range> parse(mlir::AsmParser &parser);
};

mlir::AsmPrinter &operator<<(mlir::AsmPrinter &printer,
                             const mlir::modeling::Range &range);

template <>
struct FieldParser<std::optional<mlir::modeling::Range>> {
  static FailureOr<std::optional<mlir::modeling::Range>>
  parse(mlir::AsmParser &parser);
};

mlir::AsmPrinter &operator<<(mlir::AsmPrinter &printer,
                             const std::optional<mlir::modeling::Range> &range);

template <>
struct FieldParser<mlir::modeling::MultidimensionalRange> {
  static FailureOr<mlir::modeling::MultidimensionalRange>
  parse(mlir::AsmParser &parser);
};

mlir::AsmPrinter &
operator<<(mlir::AsmPrinter &printer,
           const mlir::modeling::MultidimensionalRange &range);

template <>
struct FieldParser<std::optional<mlir::modeling::MultidimensionalRange>> {
  static FailureOr<std::optional<mlir::modeling::MultidimensionalRange>>
  parse(mlir::AsmParser &parser);
};

mlir::AsmPrinter &
operator<<(mlir::AsmPrinter &printer,
           const std::optional<mlir::modeling::MultidimensionalRange> &range);
} // namespace mlir

#endif // MARCO_DIALECT_MODELING_IR_ATTRIBUTES_H
