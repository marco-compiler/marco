#include "marco/Dialect/Modeling/IR/Properties.h"
#include "marco/Dialect/Modeling/IR/Attributes.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Builders.h"

namespace mlir::modeling {
//===-------------------------------------------------------------------===//
// IndexSet
//===-------------------------------------------------------------------===//

mlir::LogicalResult setPropertiesFromAttribute(
    IndexSet &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr);

  if (!arrayAttr) {
    emitError() << "expected ArrayAttr to set IndexSet property";
    return mlir::failure();
  }

  for (MultidimensionalRangeAttr rangeAttr :
       arrayAttr.getAsRange<MultidimensionalRangeAttr>()) {
    prop += rangeAttr.getValue();
  }

  return mlir::success();
}

mlir::ArrayAttr getPropertiesAsAttribute(mlir::MLIRContext *context,
                                         const IndexSet &prop) {
  IndexSet canonical = prop.getCanonicalRepresentation();
  llvm::SmallVector<mlir::Attribute> rangeAttrs;

  for (const MultidimensionalRange &range :
       llvm::make_range(canonical.rangesBegin(), canonical.rangesEnd())) {
    rangeAttrs.push_back(MultidimensionalRangeAttr::get(context, range));
  }

  mlir::Builder builder(context);
  return builder.getArrayAttr(rangeAttrs);
}

llvm::hash_code computeHash(const IndexSet &prop) { return hash_value(prop); }

mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         IndexSet &prop) {
  uint64_t rank, numOfRanges;

  if (mlir::failed(reader.readVarInt(rank))) {
    return mlir::failure();
  }

  if (mlir::failed(reader.readVarInt(numOfRanges))) {
    return mlir::failure();
  }

  llvm::SmallVector<Range, 3> ranges;

  for (uint64_t i = 0; i < numOfRanges; ++i) {
    ranges.clear();

    for (uint64_t dim = 0; dim < rank; ++dim) {
      int64_t begin, end;

      if (mlir::failed(reader.readSignedVarInt(begin))) {
        return mlir::failure();
      }

      if (mlir::failed(reader.readSignedVarInt(end))) {
        return mlir::failure();
      }

      ranges.emplace_back(begin, end);
    }

    prop += MultidimensionalRange(ranges);
  }

  return mlir::success();
}

void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         const IndexSet &prop) {
  IndexSet canonical = prop.getCanonicalRepresentation();
  llvm::SmallVector<MultidimensionalRange> ranges;

  for (const MultidimensionalRange &range :
       llvm::make_range(canonical.rangesBegin(), canonical.rangesEnd())) {
    ranges.push_back(range);
  }

  size_t rank = prop.rank();
  writer.writeVarInt(rank);
  writer.writeVarInt(ranges.size());

  for (const MultidimensionalRange &range : ranges) {
    for (size_t dim = 0; dim < rank; ++dim) {
      writer.writeSignedVarInt(range[dim].getBegin());
      writer.writeSignedVarInt(range[dim].getEnd());
    }
  }
}

mlir::LogicalResult parse(mlir::OpAsmParser &parser, IndexSet &prop) {
  if (parser.parseLBrace()) {
    return mlir::failure();
  }

  llvm::SmallVector<Range> ranges;

  if (mlir::failed(parser.parseOptionalRBrace())) {
    do {
      ranges.clear();

      if (parser.parseLSquare()) {
        return mlir::failure();
      }

      do {
        int64_t begin, end;

        if (parser.parseInteger(begin) || parser.parseComma() ||
            parser.parseInteger(end) || parser.parseRSquare()) {
          return mlir::failure();
        }

        ranges.emplace_back(begin, end + 1);
      } while (mlir::succeeded(parser.parseOptionalLSquare()));

      prop += MultidimensionalRange(ranges);
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRBrace()) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void print(mlir::OpAsmPrinter &printer, const IndexSet &prop) {
  IndexSet canonical = prop.getCanonicalRepresentation();
  printer << "{";

  auto rangeFn = [&](const MultidimensionalRange &range) {
    for (size_t dim = 0, rank = range.rank(); dim < rank; ++dim) {
      printer << "[" << range[dim].getBegin() << ","
              << (range[dim].getEnd() - 1) << "]";
    }
  };

  auto betweenFn = [&]() { printer << ","; };

  llvm::interleave(canonical.rangesBegin(), canonical.rangesEnd(), rangeFn,
                   betweenFn);

  printer << "}";
}

//===-------------------------------------------------------------------===//
// IndexSetsList
//===-------------------------------------------------------------------===//

mlir::LogicalResult setPropertiesFromAttribute(
    IndexSetsList &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr);

  if (!arrayAttr) {
    emitError() << "expected ArrayAttr to set IndexSetsList property";
    return mlir::failure();
  }

  for (mlir::Attribute elementAttr : arrayAttr) {
    if (mlir::failed(setPropertiesFromAttribute(prop.emplace_back(),
                                                elementAttr, emitError))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::ArrayAttr getPropertiesAsAttribute(mlir::MLIRContext *context,
                                         const IndexSetsList &prop) {
  llvm::SmallVector<mlir::Attribute> variableAttrs;

  for (const IndexSet &indexSet : prop) {
    variableAttrs.push_back(getPropertiesAsAttribute(context, indexSet));
  }

  mlir::Builder builder(context);
  return builder.getArrayAttr(variableAttrs);
}

llvm::hash_code computeHash(const IndexSetsList &prop) {
  return llvm::hash_combine_range(prop.begin(), prop.end());
}

mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         IndexSetsList &prop) {
  uint64_t numOfElements;

  if (mlir::failed(reader.readVarInt(numOfElements))) {
    return mlir::failure();
  }

  for (uint64_t i = 0; i < numOfElements; ++i) {
    if (mlir::failed(readFromMlirBytecode(reader, prop.emplace_back()))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         IndexSetsList &prop) {
  writer.writeVarInt(prop.size());

  for (const IndexSet &indexSet : prop) {
    writeToMlirBytecode(writer, indexSet);
  }
}

mlir::LogicalResult parse(mlir::OpAsmParser &parser, IndexSetsList &prop) {
  if (parser.parseLSquare()) {
    return mlir::failure();
  }

  if (mlir::failed(parser.parseOptionalRSquare())) {
    IndexSet indices;

    do {
      if (mlir::failed(parse(parser, indices))) {
        return mlir::failure();
      }

      prop.push_back(std::move(indices));
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare()) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void print(mlir::OpAsmPrinter &printer, const IndexSetsList &prop) {
  printer << "[";

  llvm::interleaveComma(
      prop, printer, [&](const IndexSet &indices) { print(printer, indices); });

  printer << "]";
}
} // namespace mlir::modeling
