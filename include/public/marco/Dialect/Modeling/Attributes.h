#ifndef MARCO_DIALECTS_MODELING_ATTRIBUTES_H
#define MARCO_DIALECTS_MODELING_ATTRIBUTES_H

#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/StorageUniquer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Hashing.h"

namespace mlir::modeling
{
  using Point = ::marco::modeling::Point;
  using Range = ::marco::modeling::Range;
  using MultidimensionalRange = ::marco::modeling::MultidimensionalRange;
  using IndexSet = ::marco::modeling::IndexSet;
}

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Modeling/ModelingAttributes.h.inc"

namespace mlir
{
  template<>
  struct FieldParser<mlir::modeling::Range>
  {
    static FailureOr<mlir::modeling::Range>
    parse(mlir::AsmParser& parser);
  };

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer, const mlir::modeling::Range& range);

  template<>
  struct FieldParser<llvm::Optional<mlir::modeling::Range>>
  {
    static FailureOr<llvm::Optional<mlir::modeling::Range>>
    parse(mlir::AsmParser& parser);
  };

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer,
      const llvm::Optional<mlir::modeling::Range>& range);

  template<>
  struct FieldParser<mlir::modeling::MultidimensionalRange>
  {
    static FailureOr<mlir::modeling::MultidimensionalRange>
    parse(mlir::AsmParser& parser);
  };

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer,
      const mlir::modeling::MultidimensionalRange& range);

  template<>
  struct FieldParser<llvm::Optional<mlir::modeling::MultidimensionalRange>>
  {
    static FailureOr<llvm::Optional<mlir::modeling::MultidimensionalRange>>
    parse(mlir::AsmParser& parser);
  };

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer,
      const llvm::Optional<mlir::modeling::MultidimensionalRange>& range);

  template<>
  struct FieldParser<mlir::modeling::IndexSet>
  {
    static FailureOr<mlir::modeling::IndexSet>
    parse(mlir::AsmParser& parser);
  };

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer,
      const mlir::modeling::IndexSet& indexSet);

  template<>
  struct FieldParser<llvm::Optional<mlir::modeling::IndexSet>>
  {
    static FailureOr<llvm::Optional<mlir::modeling::IndexSet>>
    parse(mlir::AsmParser& parser);
  };

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer,
      const llvm::Optional<mlir::modeling::IndexSet>& indexSet);
}

#endif // MARCO_DIALECTS_MODELING_ATTRIBUTES_H
