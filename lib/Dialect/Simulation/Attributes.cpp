#include "marco/Dialect/Simulation/Attributes.h"
#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::simulation;

//===---------------------------------------------------------------------===//
// Tablegen attribute definitions
//===---------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Simulation/SimulationAttributes.cpp.inc"

//===---------------------------------------------------------------------===//
// SimulationDialect
//===---------------------------------------------------------------------===//

namespace mlir::simulation
{
  void SimulationDialect::registerAttributes()
  {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "marco/Dialect/Simulation/SimulationAttributes.cpp.inc"
        >();
  }
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VariableAttr

namespace mlir::simulation
{
  int64_t VariableAttr::getRank() const
  {
    return getDimensions().size();
  }
}

//===----------------------------------------------------------------------===//
// MultidimensionalRangeAttr

namespace mlir::simulation
{
  mlir::Attribute MultidimensionalRangeAttr::parse(
      mlir::AsmParser& parser, mlir::Type type)
  {
    llvm::SmallVector<std::pair<int64_t, int64_t>> ranges;

    if (parser.parseLess()) {
      return mlir::Attribute();
    }

    if (mlir::failed(parser.parseOptionalGreater())) {
      do {
        int64_t begin, end;

        if (parser.parseLSquare() ||
            parser.parseInteger(begin) ||
            parser.parseComma() ||
            parser.parseInteger(end) ||
            parser.parseRSquare()) {
          return mlir::Attribute();
        }

        ranges.emplace_back(begin, end);
      } while (parser.parseOptionalComma().succeeded());

      if (parser.parseGreater()) {
        return mlir::Attribute();
      }
    }

    return MultidimensionalRangeAttr::get(parser.getContext(), ranges);
  }

  void MultidimensionalRangeAttr::print(mlir::AsmPrinter& printer) const
  {
    printer << "<";

    for (const auto& range : llvm::enumerate(getRanges())) {
      if (range.index() != 0) {
        printer << ", ";
      }

      printer << "["
              << range.value().first << ", "
              << range.value().second
              << "]";
    }

    printer << ">";
  }
}
