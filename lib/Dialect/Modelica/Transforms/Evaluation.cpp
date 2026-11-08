#include "marco/Dialect/Modelica/Transforms/Evaluation.h"
#include "marco/Dialect/Modelica/IR/ConstantFolding.h"
#include "marco/Dialect/Modelica/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace ::mlir::modelica;

namespace mlir::modelica {
Evaluator::Evaluator(Hook hook) : hook(std::move(hook)) {}

void Evaluator::setValue(mlir::Value value, mlir::Attribute attr) {
  values[value] = attr;
}

mlir::Attribute Evaluator::evaluate(mlir::Value value) {
  if (auto it = values.find(value); it != values.end()) {
    return it->second;
  }

  if (auto it = cache.find(value); it != cache.end()) {
    return it->second;
  }

  mlir::Attribute result;

  if (std::optional<mlir::Attribute> hookResult =
          hook ? hook(value) : std::nullopt) {
    result = *hookResult;
  } else if (auto opResult = mlir::dyn_cast<mlir::OpResult>(value)) {
    result = evaluateResult(opResult);
  }

  cache[value] = result;
  return result;
}

mlir::Attribute Evaluator::evaluateResult(mlir::OpResult result) {
  mlir::Operation *op = result.getOwner();

  if (auto constantOp = mlir::dyn_cast<ConstantOp>(op)) {
    return constantOp.getValue();
  }

  if (auto subscriptOp = mlir::dyn_cast<SubscriptOp>(op)) {
    mlir::Attribute array = evaluate(subscriptOp.getArray());

    if (!array) {
      return {};
    }

    llvm::SmallVector<mlir::Attribute> subscripts;
    auto indices = subscriptOp.getIndices();
    size_t indexPos = 0;

    for (bool isColon : subscriptOp.getColons()) {
      if (isColon) {
        subscripts.push_back(nullptr);
        continue;
      }

      mlir::Attribute index = evaluate(indices[indexPos++]);

      if (!index) {
        return {};
      }

      subscripts.push_back(index);
    }

    return subscriptConstant(array, subscripts);
  }

  if (auto ifExprOp = mlir::dyn_cast<IfExprOp>(op)) {
    std::optional<bool> condition =
        getBooleanValue(evaluate(ifExprOp.getCondition()));

    if (!condition) {
      return {};
    }

    mlir::Attribute value = evaluateRegion(
        *condition ? ifExprOp.getThenRegion() : ifExprOp.getElseRegion(), {});

    return value ? convertConstant(value, ifExprOp.getType()) : nullptr;
  }

  // Operations whose result shape depends on the values of the operands,
  // which can't be folded when the shape is not known.
  if (auto rangeOp = mlir::dyn_cast<RangeOp>(op)) {
    mlir::Attribute start = evaluate(rangeOp.getStart());
    mlir::Attribute stop = evaluate(rangeOp.getStop());
    mlir::Attribute step =
        rangeOp.getStep() ? evaluate(rangeOp.getStep()) : nullptr;

    if (!start || !stop || (rangeOp.getStep() && !step)) {
      return {};
    }

    std::optional<int64_t> size = getRangeSize(start, stop, step);

    if (!size) {
      return {};
    }

    mlir::Type elementType = getElementTypeOrSelf(rangeOp.getType());
    llvm::SmallVector<mlir::Attribute> elements;

    if (mlir::isa<IntegerType>(elementType)) {
      int64_t first = *getIntegerValue(start);
      int64_t increment = step ? *getIntegerValue(step) : 1;

      for (int64_t i = 0; i < *size; ++i) {
        elements.push_back(IntegerAttr::get(op->getContext(), elementType,
                                            first + i * increment,
                                            mlir::StringAttr()));
      }
    } else if (mlir::isa<RealType>(elementType)) {
      double first = *getRealValue(start);
      double increment = step ? *getRealValue(step) : 1;

      for (int64_t i = 0; i < *size; ++i) {
        elements.push_back(getNumericAttr(
            elementType, first + static_cast<double>(i) * increment));
      }
    } else if (mlir::isa<BooleanType>(elementType)) {
      bool first = *getBooleanValue(start);

      for (int64_t i = 0; i < *size; ++i) {
        elements.push_back(
            BooleanAttr::get(op->getContext(), elementType, first || i > 0));
      }
    } else {
      return {};
    }

    return ArrayValueAttr::get(
        op->getContext(), mlir::RankedTensorType::get({*size}, elementType),
        elements);
  }

  if (mlir::isa<FillOp, ZerosOp, OnesOp>(op)) {
    mlir::Attribute value;
    mlir::ValueRange sizes;

    if (auto fillOp = mlir::dyn_cast<FillOp>(op)) {
      value = evaluate(fillOp.getValue());
      sizes = fillOp.getSizes();
    } else {
      value = IntegerAttr::get(op->getContext(),
                               mlir::isa<OnesOp>(op) ? 1 : 0);
      sizes = op->getOperands();
    }

    if (!value) {
      return {};
    }

    llvm::SmallVector<int64_t> shape;

    for (mlir::Value size : sizes) {
      std::optional<int64_t> sizeValue = getIntegerValue(evaluate(size));

      if (!sizeValue || *sizeValue < 0) {
        return {};
      }

      shape.push_back(*sizeValue);
    }

    auto valueType = mlir::cast<mlir::TypedAttr>(value).getType();
    llvm::ArrayRef<int64_t> valueShape = modelica::getShape(valueType);
    shape.append(valueShape.begin(), valueShape.end());

    llvm::SmallVector<mlir::Attribute> valueElements =
        getConstantElements(value);

    int64_t numElements = 1;

    for (int64_t size : llvm::drop_end(shape, valueShape.size())) {
      numElements *= size;
    }

    llvm::SmallVector<mlir::Attribute> elements;

    for (int64_t i = 0; i < numElements; ++i) {
      elements.append(valueElements.begin(), valueElements.end());
    }

    return ArrayValueAttr::get(
        op->getContext(),
        mlir::RankedTensorType::get(shape, getElementTypeOrSelf(valueType)),
        elements);
  }

  if (mlir::isa<ArrayOp, CatOp, PromoteOp, VectorOp, MatrixOp>(op)) {
    return evaluateArrayConstruction(op);
  }

  if (mlir::isa<ArrayForOp, ReductionOp>(op)) {
    llvm::SmallVector<llvm::SmallVector<mlir::Attribute>> ranges;

    for (mlir::Value range : op->getOperands()) {
      mlir::Attribute rangeAttr = evaluate(range);

      if (!rangeAttr) {
        return {};
      }

      ranges.push_back(getConstantElements(rangeAttr));
    }

    mlir::Region &body = op->getRegion(0);
    llvm::SmallVector<mlir::Attribute> elements;
    llvm::SmallVector<size_t> counters(ranges.size(), 0);

    if (llvm::any_of(ranges, [](const auto &range) { return range.empty(); })) {
      if (mlir::isa<ReductionOp>(op)) {
        // Reductions over empty ranges are not folded.
        return {};
      }
    } else {
      // The first iterator varies along the innermost dimension.
      while (true) {
        llvm::SmallVector<mlir::Attribute> arguments;

        for (size_t i = 0; i < ranges.size(); ++i) {
          arguments.push_back(ranges[i][counters[i]]);
        }

        mlir::Attribute element = evaluateRegion(body, arguments);

        if (!element) {
          return {};
        }

        elements.push_back(element);

        size_t pos = 0;

        while (pos < counters.size() && ++counters[pos] == ranges[pos].size()) {
          counters[pos++] = 0;
        }

        if (pos == counters.size()) {
          break;
        }
      }
    }

    if (auto reductionOp = mlir::dyn_cast<ReductionOp>(op)) {
      mlir::Type type = reductionOp.getType();

      if (!isNumericType(type)) {
        return {};
      }

      bool isInteger = mlir::isa<IntegerType>(type);
      std::optional<double> accumulator;

      for (mlir::Attribute element : elements) {
        std::optional<double> value = getRealValue(element);

        if (!value || hasUnit(element)) {
          return {};
        }

        if (!accumulator) {
          accumulator = *value;
          continue;
        }

        switch (reductionOp.getKind()) {
        case ReductionKind::sum:
          *accumulator += *value;
          break;
        case ReductionKind::product:
          *accumulator *= *value;
          break;
        case ReductionKind::min:
          *accumulator = std::min(*accumulator, *value);
          break;
        case ReductionKind::max:
          *accumulator = std::max(*accumulator, *value);
          break;
        }
      }

      if (isInteger &&
          std::abs(*accumulator) > static_cast<double>(1LL << 52)) {
        return {};
      }

      return getNumericAttr(type, *accumulator);
    }

    // Array constructor with iterators.
    llvm::SmallVector<int64_t> shape;

    for (const auto &range : llvm::reverse(ranges)) {
      shape.push_back(range.size());
    }

    mlir::Type elementType = getElementTypeOrSelf(result.getType());
    llvm::SmallVector<mlir::Attribute> flattened;

    if (!elements.empty()) {
      auto elementShape = modelica::getShape(
          mlir::cast<mlir::TypedAttr>(elements.front()).getType());

      shape.append(elementShape.begin(), elementShape.end());
    }

    for (mlir::Attribute element : elements) {
      for (mlir::Attribute scalar : getConstantElements(element)) {
        mlir::Attribute converted = convertScalarAttr(scalar, elementType);

        if (!converted) {
          return {};
        }

        flattened.push_back(converted);
      }
    }

    return ArrayValueAttr::get(op->getContext(),
                               mlir::RankedTensorType::get(shape, elementType),
                               flattened);
  }

  if (op->getNumRegions() != 0) {
    return {};
  }

  // Operands whose value is unknown are passed as null attributes, as the
  // result may not depend on them (e.g., the size of a statically shaped
  // array).
  llvm::SmallVector<mlir::Attribute> operands;

  for (mlir::Value operand : op->getOperands()) {
    operands.push_back(evaluate(operand));
  }

  llvm::SmallVector<mlir::OpFoldResult> foldResults;

  if (mlir::failed(op->fold(operands, foldResults)) ||
      foldResults.size() != op->getNumResults()) {
    return {};
  }

  mlir::OpFoldResult foldResult = foldResults[result.getResultNumber()];

  if (auto attr = mlir::dyn_cast<mlir::Attribute>(foldResult)) {
    return attr;
  }

  auto foldedValue = mlir::cast<mlir::Value>(foldResult);

  if (foldedValue == result) {
    return {};
  }

  return evaluate(foldedValue);
}

mlir::Attribute Evaluator::evaluateArrayConstruction(mlir::Operation *op) {
  mlir::MLIRContext *context = op->getContext();
  mlir::Type elementType = getElementTypeOrSelf(op->getResult(0).getType());

  auto getArray = [&](mlir::Attribute attr, llvm::SmallVectorImpl<int64_t> &shape,
                      llvm::SmallVectorImpl<mlir::Attribute> &elements) {
    auto typedAttr = mlir::cast<mlir::TypedAttr>(attr);
    llvm::ArrayRef<int64_t> attrShape = modelica::getShape(typedAttr.getType());
    shape.assign(attrShape.begin(), attrShape.end());

    for (mlir::Attribute element : getConstantElements(attr)) {
      mlir::Attribute converted = convertScalarAttr(element, elementType);

      if (!converted) {
        return false;
      }

      elements.push_back(converted);
    }

    return true;
  };

  auto makeArray = [&](llvm::ArrayRef<int64_t> shape,
                       llvm::ArrayRef<mlir::Attribute> elements)
      -> mlir::Attribute {
    return ArrayValueAttr::get(
        context, mlir::RankedTensorType::get(shape, elementType), elements);
  };

  llvm::SmallVector<mlir::Attribute> operands;

  for (mlir::Value operand : op->getOperands()) {
    mlir::Attribute value = evaluate(operand);

    if (!value) {
      return {};
    }

    operands.push_back(value);
  }

  if (mlir::isa<ArrayOp>(op)) {
    llvm::SmallVector<int64_t> resultShape = {
        static_cast<int64_t>(operands.size())};

    llvm::SmallVector<mlir::Attribute> elements;
    std::optional<llvm::SmallVector<int64_t>> elementShape;

    for (mlir::Attribute operand : operands) {
      llvm::SmallVector<int64_t> shape;

      if (!getArray(operand, shape, elements) ||
          (elementShape && *elementShape != shape)) {
        return {};
      }

      elementShape = shape;
    }

    if (elementShape) {
      resultShape.append(elementShape->begin(), elementShape->end());
    }

    return makeArray(resultShape, elements);
  }

  if (auto catOp = mlir::dyn_cast<CatOp>(op)) {
    std::optional<int64_t> dim = getIntegerValue(operands.front());

    if (!dim || *dim < 0) {
      return {};
    }

    llvm::SmallVector<llvm::SmallVector<int64_t>> shapes;
    llvm::SmallVector<llvm::SmallVector<mlir::Attribute>> arrays;

    for (mlir::Attribute operand : llvm::drop_begin(operands)) {
      if (!getArray(operand, shapes.emplace_back(), arrays.emplace_back()) ||
          static_cast<int64_t>(shapes.back().size()) <= *dim ||
          shapes.back().size() != shapes.front().size()) {
        return {};
      }
    }

    if (shapes.empty()) {
      return {};
    }

    size_t d = *dim;
    llvm::SmallVector<int64_t> resultShape(shapes.front());
    resultShape[d] = 0;
    int64_t outer = 1, inner = 1;

    for (size_t i = 0; i < resultShape.size(); ++i) {
      if (i < d) {
        outer *= resultShape[i];
      } else if (i > d) {
        inner *= resultShape[i];
      }
    }

    for (const auto &shape : shapes) {
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i != d && shape[i] != shapes.front()[i]) {
          return {};
        }
      }

      resultShape[d] += shape[d];
    }

    llvm::SmallVector<mlir::Attribute> elements;

    for (int64_t o = 0; o < outer; ++o) {
      for (auto [shape, array] : llvm::zip_equal(shapes, arrays)) {
        int64_t blockSize = shape[d] * inner;
        auto block = llvm::ArrayRef(array).slice(o * blockSize, blockSize);
        elements.append(block.begin(), block.end());
      }
    }

    return makeArray(resultShape, elements);
  }

  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<mlir::Attribute> elements;

  if (!getArray(operands.front(), shape, elements)) {
    return {};
  }

  if (auto promoteOp = mlir::dyn_cast<PromoteOp>(op)) {
    while (static_cast<int64_t>(shape.size()) < promoteOp.getRank()) {
      shape.push_back(1);
    }

    return makeArray(shape, elements);
  }

  if (mlir::isa<VectorOp>(op)) {
    if (llvm::count_if(shape, [](int64_t size) { return size > 1; }) > 1) {
      return {};
    }

    return makeArray({static_cast<int64_t>(elements.size())}, elements);
  }

  // Matrix conversion.
  while (shape.size() < 2) {
    shape.push_back(1);
  }

  if (llvm::any_of(llvm::drop_begin(shape, 2),
                   [](int64_t size) { return size != 1; })) {
    return {};
  }

  shape.resize(2);
  return makeArray(shape, elements);
}

mlir::Attribute
Evaluator::evaluateRegion(mlir::Region &region,
                          llvm::ArrayRef<mlir::Attribute> arguments) {
  if (region.empty()) {
    return {};
  }

  auto yieldOp = mlir::dyn_cast<YieldOp>(region.front().getTerminator());

  if (!yieldOp || yieldOp.getValues().size() != 1) {
    return {};
  }

  // The values computed within the region depend on its arguments, and thus
  // are evaluated by a nested evaluator.
  Evaluator nested(hook);
  nested.values = values;

  for (auto [argument, value] :
       llvm::zip_equal(region.getArguments(), arguments)) {
    nested.setValue(argument, value);
  }

  return nested.evaluate(yieldOp.getValues()[0]);
}

std::optional<int64_t> getIndexValue(mlir::Attribute index) {
  if (!index) {
    return std::nullopt;
  }

  if (auto integerAttr = mlir::dyn_cast<IntegerAttr>(index)) {
    return integerAttr.getValue();
  }

  if (auto booleanAttr = mlir::dyn_cast<BooleanAttr>(index)) {
    return booleanAttr.getValue() ? 1 : 0;
  }

  if (auto enumAttr = mlir::dyn_cast<EnumLiteralAttr>(index)) {
    return enumAttr.getOrdinal() - 1;
  }

  return std::nullopt;
}

mlir::Attribute subscriptConstant(mlir::Attribute array,
                                  llvm::ArrayRef<mlir::Attribute> subscripts) {
  auto arrayAttr = mlir::dyn_cast<ArrayValueAttr>(array);

  if (!arrayAttr) {
    return {};
  }

  mlir::RankedTensorType arrayType = arrayAttr.getTensorType();
  llvm::ArrayRef<int64_t> shape = arrayType.getShape();

  if (subscripts.size() > shape.size()) {
    return {};
  }

  // Positions selected along each dimension, and whether the dimension is
  // kept in the result.
  llvm::SmallVector<llvm::SmallVector<int64_t>> selected;
  llvm::SmallVector<int64_t> resultShape;

  for (size_t dim = 0; dim < shape.size(); ++dim) {
    llvm::SmallVector<int64_t> &positions = selected.emplace_back();
    mlir::Attribute subscript =
        dim < subscripts.size() ? subscripts[dim] : nullptr;

    if (!subscript) {
      for (int64_t i = 0; i < shape[dim]; ++i) {
        positions.push_back(i);
      }

      resultShape.push_back(shape[dim]);
      continue;
    }

    bool isVector = mlir::isa<ArrayValueAttr>(subscript);

    for (mlir::Attribute index : getConstantElements(subscript)) {
      std::optional<int64_t> position = getIndexValue(index);

      if (!position || *position < 0 || *position >= shape[dim]) {
        return {};
      }

      positions.push_back(*position);
    }

    if (isVector) {
      resultShape.push_back(positions.size());
    }
  }

  llvm::ArrayRef<mlir::Attribute> elements = arrayAttr.getElements();
  llvm::SmallVector<mlir::Attribute> resultElements;
  llvm::SmallVector<size_t> counters(shape.size(), 0);

  if (llvm::any_of(selected, [](const auto &positions) {
        return positions.empty();
      })) {
    return ArrayValueAttr::get(
        array.getContext(),
        mlir::RankedTensorType::get(resultShape, arrayType.getElementType()),
        {});
  }

  while (true) {
    int64_t offset = 0;

    for (size_t dim = 0; dim < shape.size(); ++dim) {
      offset = offset * shape[dim] + selected[dim][counters[dim]];
    }

    resultElements.push_back(elements[offset]);

    size_t dim = shape.size();

    while (dim > 0 && ++counters[dim - 1] == selected[dim - 1].size()) {
      counters[--dim] = 0;
    }

    if (dim == 0) {
      break;
    }
  }

  if (resultShape.empty()) {
    return resultElements.front();
  }

  return ArrayValueAttr::get(
      array.getContext(),
      mlir::RankedTensorType::get(resultShape, arrayType.getElementType()),
      resultElements);
}

mlir::Attribute convertConstant(mlir::Attribute attr, mlir::Type type) {
  auto arrayAttr = mlir::dyn_cast<ArrayValueAttr>(attr);

  if (!arrayAttr) {
    if (mlir::isa<mlir::RankedTensorType>(type)) {
      return {};
    }

    return convertScalarAttr(attr, type);
  }

  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);

  if (!tensorType || tensorType.getRank() != arrayAttr.getTensorType().getRank()) {
    return {};
  }

  for (auto [expected, actual] : llvm::zip_equal(
           tensorType.getShape(), arrayAttr.getTensorType().getShape())) {
    if (!mlir::ShapedType::isDynamic(expected) && expected != actual) {
      return {};
    }
  }

  mlir::Type elementType = tensorType.getElementType();
  llvm::SmallVector<mlir::Attribute> elements;

  for (mlir::Attribute element : arrayAttr.getElements()) {
    mlir::Attribute converted = convertScalarAttr(element, elementType);

    if (!converted) {
      return {};
    }

    elements.push_back(converted);
  }

  return ArrayValueAttr::get(
      attr.getContext(),
      mlir::RankedTensorType::get(arrayAttr.getTensorType().getShape(),
                                  elementType),
      elements);
}
} // namespace mlir::modelica
