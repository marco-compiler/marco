#include "marco/Dialect/Modelica/IR/OpUtils.h"
#include "mlir/IR/TypeUtilities.h"

using namespace ::mlir::modelica;

namespace mlir::modelica {
bool isComponentReference(mlir::Value value, bool allowDerivative) {
  mlir::Operation *op = value.getDefiningOp();

  while (op) {
    if (auto refOp = mlir::dyn_cast<RefOp>(op)) {
      return refOp.hasSimpleName();
    }

    if (auto memberOp = mlir::dyn_cast<MemberOp>(op)) {
      op = memberOp.getObject().getDefiningOp();
      continue;
    }

    if (auto subscriptOp = mlir::dyn_cast<SubscriptOp>(op)) {
      op = subscriptOp.getArray().getDefiningOp();
      continue;
    }

    if (auto derOp = mlir::dyn_cast<DerOp>(op); derOp && allowDerivative) {
      // Only the outermost operation can be a derivative.
      allowDerivative = false;
      op = derOp.getOperand().getDefiningOp();
      continue;
    }

    return false;
  }

  return false;
}
} // namespace mlir::modelica

namespace mlir::modelica::detail {
mlir::FailureOr<mlir::ValueRange> getYieldedValues(mlir::Operation *op,
                                                   mlir::Region &region,
                                                   llvm::StringRef regionName,
                                                   size_t minValues,
                                                   size_t maxValues) {
  if (region.empty()) {
    return op->emitOpError() << regionName << " region must not be empty";
  }

  auto yieldOp = mlir::dyn_cast<YieldOp>(region.front().getTerminator());

  if (!yieldOp) {
    return op->emitOpError()
           << regionName << " region must be terminated by 'modelica.yield'";
  }

  size_t numValues = yieldOp.getValues().size();

  if (numValues < minValues || numValues > maxValues) {
    auto diag = op->emitOpError() << regionName << " region must yield ";

    if (minValues == maxValues) {
      diag << minValues;
    } else {
      diag << "between " << minValues << " and " << maxValues;
    }

    diag << " values, but yields " << numValues;
    return diag;
  }

  return mlir::ValueRange(yieldOp.getValues());
}

mlir::LogicalResult verifyCondition(mlir::Operation *op, mlir::Region &region,
                                    bool allowVector) {
  auto values = getYieldedValues(op, region, "condition", 1, 1);

  if (mlir::failed(values)) {
    return mlir::failure();
  }

  mlir::Type type = (*values)[0].getType();
  bool isBoolean = mlir::isa<BooleanType>(type);

  if (allowVector) {
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
      isBoolean = tensorType.getRank() == 1 &&
                  mlir::isa<BooleanType>(tensorType.getElementType());
    }
  }

  if (!isBoolean) {
    return op->emitOpError("condition must be ")
           << (allowVector ? "a Boolean scalar or vector" : "a Boolean scalar")
           << ", but got " << type;
  }

  return mlir::success();
}

mlir::LogicalResult verifyComponentReference(mlir::Operation *op,
                                             mlir::Value value,
                                             llvm::StringRef description,
                                             bool allowDerivative) {
  if (!isComponentReference(value, allowDerivative)) {
    return op->emitOpError() << description << " must be a component reference";
  }

  return mlir::success();
}

void printAttributes(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                     llvm::ArrayRef<llvm::StringRef> elidedAttrs,
                     bool withKeyword) {
  mlir::NamedAttrList attrs(op->getDiscardableAttrDictionary());

  if (auto properties = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
          op->getPropertiesAsAttribute())) {
    for (mlir::NamedAttribute attr : properties) {
      attrs.push_back(attr);
    }
  }

  if (withKeyword) {
    printer.printOptionalAttrDictWithKeyword(attrs.getAttrs(), elidedAttrs);
  } else {
    printer.printOptionalAttrDict(attrs.getAttrs(), elidedAttrs);
  }
}

mlir::ParseResult parseRegionWithBlock(mlir::OpAsmParser &parser,
                                       mlir::Region &region) {
  if (parser.parseRegion(region)) {
    return mlir::failure();
  }

  if (region.empty()) {
    region.emplaceBlock();
  }

  return mlir::success();
}

mlir::ParseResult parseOptionalRegion(mlir::OpAsmParser &parser,
                                      mlir::Region &region) {
  mlir::OptionalParseResult parseResult = parser.parseOptionalRegion(region);

  if (parseResult.has_value() && mlir::failed(*parseResult)) {
    return mlir::failure();
  }

  return mlir::success();
}

void printOptionalRegion(mlir::OpAsmPrinter &printer, mlir::Region &region,
                         llvm::StringRef keyword) {
  if (region.empty()) {
    return;
  }

  printer << " ";

  if (!keyword.empty()) {
    printer << keyword << " ";
  }

  printer.printRegion(region, false);
}

mlir::LogicalResult verifyModificationRegion(mlir::Operation *op,
                                             mlir::Region &region) {
  if (region.empty()) {
    return mlir::success();
  }

  if (llvm::count_if(region.front(), [](mlir::Operation &nested) {
        return mlir::isa<BindOp>(nested);
      }) > 1) {
    return op->emitOpError("modification can't have more than one value");
  }

  return mlir::success();
}
} // namespace mlir::modelica::detail
