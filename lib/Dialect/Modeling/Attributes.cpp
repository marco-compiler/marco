#include "marco/Dialect/Modeling/Attributes.h"
#include "marco/Dialect/Modeling/ModelingDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modeling;

namespace mlir
{
  mlir::FailureOr<mlir::modeling::Range>
  FieldParser<mlir::modeling::Range>::parse(mlir::AsmParser& parser)
  {
    int64_t beginValue;
    int64_t endValue;

    if (parser.parseLSquare() ||
        parser.parseInteger(beginValue) ||
        parser.parseComma() ||
        parser.parseInteger(endValue) ||
        parser.parseRSquare()) {
      return mlir::failure();
    }

    return mlir::modeling::Range(beginValue, endValue + 1);
  }

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer, const Range& range)
  {
    return printer << "[" << range.getBegin() << ","
                   << (range.getEnd() - 1) << "]";
  }

  mlir::FailureOr<std::optional<mlir::modeling::Range>>
  FieldParser<std::optional<mlir::modeling::Range>>::parse(
      mlir::AsmParser& parser)
  {
    int64_t beginValue;
    int64_t endValue;

    if (parser.parseOptionalLSquare()) {
      return std::optional<mlir::modeling::Range>(std::nullopt);
    }

    if (parser.parseInteger(beginValue) ||
        parser.parseComma() ||
        parser.parseInteger(endValue) ||
        parser.parseRSquare()) {
      return mlir::failure();
    }

    return std::optional(mlir::modeling::Range(beginValue, endValue + 1));
  }

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer, const std::optional<Range>& range)
  {
    if (range) {
      printer << *range;
    }

    return printer;
  }

  FailureOr<MultidimensionalRange>
  FieldParser<MultidimensionalRange>::parse(mlir::AsmParser& parser)
  {
    llvm::SmallVector<Range, 3> ranges;

    mlir::FailureOr<mlir::modeling::Range> range =
        FieldParser<mlir::modeling::Range>::parse(parser);

    if (mlir::failed(range)) {
      return mlir::failure();
    }

    ranges.push_back(*range);

    while (true) {
      mlir::FailureOr<std::optional<mlir::modeling::Range>> optionalRange =
          FieldParser<std::optional<mlir::modeling::Range>>::parse(parser);

      if (mlir::failed(optionalRange)) {
        return mlir::failure();
      }

      if (*optionalRange) {
        ranges.push_back(**optionalRange);
      } else {
        break;
      }
    }

    return MultidimensionalRange(ranges);
  }

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer,
      const mlir::modeling::MultidimensionalRange& range)
  {
    for (unsigned int i = 0, e = range.rank(); i < e; ++i) {
      printer << range[i];
    }

    return printer;
  }

  FailureOr<std::optional<mlir::modeling::MultidimensionalRange>>
  FieldParser<std::optional<mlir::modeling::MultidimensionalRange>>::parse(
      mlir::AsmParser& parser)
  {
    llvm::SmallVector<Range, 3> ranges;

    mlir::FailureOr<std::optional<mlir::modeling::Range>> range =
        FieldParser<std::optional<mlir::modeling::Range>>::parse(parser);

    if (mlir::failed(range)) {
      return mlir::failure();
    }

    if (!(*range)) {
      return std::optional<mlir::modeling::MultidimensionalRange>(std::nullopt);
    }

    ranges.push_back(**range);

    while (true) {
      mlir::FailureOr<std::optional<mlir::modeling::Range>> optionalRange =
          FieldParser<std::optional<mlir::modeling::Range>>::parse(parser);

      if (mlir::failed(optionalRange)) {
        return mlir::failure();
      }

      if (*optionalRange) {
        ranges.push_back(**optionalRange);
      } else {
        break;
      }
    }

    return std::optional(mlir::modeling::MultidimensionalRange(ranges));
  }

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer,
      const std::optional<mlir::modeling::MultidimensionalRange>& range)
  {
    if (range) {
      printer << *range;
    }

    return printer;
  }

  FailureOr<IndexSet> FieldParser<IndexSet>::parse(mlir::AsmParser& parser)
  {
    IndexSet result;

    if (parser.parseLBrace()) {
      return mlir::failure();
    }

    if (mlir::succeeded(parser.parseOptionalRBrace())) {
      return result;
    }

    do {
      mlir::FailureOr<mlir::modeling::MultidimensionalRange> range =
          FieldParser<mlir::modeling::MultidimensionalRange>::parse(parser);

      if (mlir::failed(range)) {
        return mlir::failure();
      }

      result += *range;
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRBrace()) {
      return mlir::failure();
    }

    return result;
  }

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer, const IndexSet& indexSet)
  {
    printer << "{";

    llvm::interleave(
        indexSet.rangesBegin(), indexSet.rangesEnd(),
        [&](const MultidimensionalRange& range) {
          printer << range;
        },
        [&]() {
          printer << ", ";
        });

    printer << "}";
    return printer;
  }

  FailureOr<std::optional<IndexSet>>
  FieldParser<std::optional<IndexSet>>::parse(mlir::AsmParser& parser)
  {
    IndexSet result;

    if (mlir::failed(parser.parseOptionalLBrace())) {
      return std::optional<IndexSet>(std::nullopt);
    }

    if (mlir::succeeded(parser.parseOptionalRBrace())) {
      return std::optional(result);
    }

    do {
      mlir::FailureOr<mlir::modeling::MultidimensionalRange> range =
          FieldParser<mlir::modeling::MultidimensionalRange>::parse(parser);

      if (mlir::failed(range)) {
        return mlir::failure();
      }

      result += *range;
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRBrace()) {
      return mlir::failure();
    }

    return std::optional(result);
  }

  mlir::AsmPrinter& operator<<(
      mlir::AsmPrinter& printer, const std::optional<IndexSet>& indexSet)
  {
    if (indexSet) {
      printer << *indexSet;
    }

    return printer;
  }
}

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Modeling/ModelingAttributes.cpp.inc"

//===---------------------------------------------------------------------===//
// ModelingDialect
//===---------------------------------------------------------------------===//

namespace mlir::modeling
{
  void ModelingDialect::registerAttributes()
  {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "marco/Dialect/Modeling/ModelingAttributes.cpp.inc"
        >();
  }
}
