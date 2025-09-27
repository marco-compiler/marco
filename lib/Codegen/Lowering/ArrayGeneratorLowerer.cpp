#include "marco/Codegen/Lowering/BaseModelica/ArrayGeneratorLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

namespace marco::codegen::lowering::bmodelica {
ArrayGeneratorLowerer::ArrayGeneratorLowerer(BridgeInterface *bridge)
    : Lowerer(bridge) {}

std::optional<Results>
ArrayGeneratorLowerer::lower(const ast::bmodelica::ArrayGenerator &array) {
  if (array.isa<ast::bmodelica::ArrayConstant>()) {
    return lower(*array.cast<ast::bmodelica::ArrayConstant>());
  } else if (array.isa<ast::bmodelica::ArrayForGenerator>()) {
    return lower(*array.cast<ast::bmodelica::ArrayForGenerator>());
  } else {
    llvm_unreachable("Unknown type of array generator");
  }
}

void ArrayGeneratorLowerer::computeShape(
    const ast::bmodelica::ArrayGenerator &array,
    llvm::SmallVectorImpl<int64_t> &outShape) {
  llvm::SmallVector<const ast::bmodelica::ArrayGenerator *> nestedArrays;
  nestedArrays.push_back(&array);

  while (!nestedArrays.empty()) {
    const ast::bmodelica::ArrayGenerator *current = nestedArrays.pop_back_val();

    if (auto constant = current->dyn_cast<ast::bmodelica::ArrayConstant>()) {
      size_t numOfChildren = constant->size();
      outShape.push_back(numOfChildren);

      if (numOfChildren > 0) {
        // TODO: check if the various children have a consistent type and size
        const ast::bmodelica::Expression *child = (*constant)[0];

        if (auto arrayChild =
                child->dyn_cast<ast::bmodelica::ArrayGenerator>()) {
          nestedArrays.push_back(arrayChild);
        }
      }

    } else if (auto forGen =
                   current->dyn_cast<ast::bmodelica::ArrayForGenerator>()) {
      unsigned inductions = forGen->getNumIndices();

      for (unsigned i = 0; i < inductions; i++) {
        const ast::bmodelica::ForIndex *ind =
            forGen->getIndex(inductions - i - 1);
        assert(ind->hasExpression());
        assert(ind->getExpression()->isa<ast::bmodelica::Operation>());
        const auto *indRange =
            ind->getExpression()->cast<ast::bmodelica::Operation>();
        assert(indRange->getOperationKind() ==
               ast::bmodelica::OperationKind::range);

        int64_t begin, end, step;

        if (indRange->getNumOfArguments() == 2) {
          assert(indRange->getArgument(0)->isa<ast::bmodelica::Constant>());
          assert(indRange->getArgument(1)->isa<ast::bmodelica::Constant>());

          begin = indRange->getArgument(0)
                      ->cast<ast::bmodelica::Constant>()
                      ->as<int64_t>();
          end = indRange->getArgument(1)
                    ->cast<ast::bmodelica::Constant>()
                    ->as<int64_t>();
          step = 1;
        } else {
          assert(indRange->getNumOfArguments() == 3);
          assert(indRange->getArgument(0)->isa<ast::bmodelica::Constant>());
          assert(indRange->getArgument(1)->isa<ast::bmodelica::Constant>());
          assert(indRange->getArgument(2)->isa<ast::bmodelica::Constant>());

          begin = indRange->getArgument(0)
                      ->cast<ast::bmodelica::Constant>()
                      ->as<int64_t>();
          end = indRange->getArgument(2)
                    ->cast<ast::bmodelica::Constant>()
                    ->as<int64_t>();
          step = indRange->getArgument(1)
                     ->cast<ast::bmodelica::Constant>()
                     ->as<int64_t>();
        }

        if (begin != 1 || step != 1) {
          assert(false && "Array for generators with step/index not equal to 1 "
                          "are not supported");
        }

        outShape.push_back(end);
      }

      if (auto arrayChild =
              forGen->getValue()->dyn_cast<ast::bmodelica::ArrayGenerator>()) {
        nestedArrays.push_back(arrayChild);
      }
    }
  }
}

bool ArrayGeneratorLowerer::lowerValues(
    const ast::bmodelica::Expression &array,
    llvm::SmallVectorImpl<mlir::Value> &outValues) {
  llvm::SmallVector<const ast::bmodelica::Expression *> s1;
  llvm::SmallVector<const ast::bmodelica::Expression *> s2;

  s1.push_back(&array);

  while (!s1.empty()) {
    const ast::bmodelica::Expression *node = s1.pop_back_val();

    if (auto arrayNode = node->dyn_cast<ast::bmodelica::ArrayConstant>()) {
      for (size_t i = 0, e = arrayNode->size(); i < e; ++i) {
        s1.push_back((*arrayNode)[i]);
      }
    } else if (auto arrayNode =
                   node->dyn_cast<ast::bmodelica::ArrayForGenerator>()) {
      // TODO: extend broadcast to arrays and avoid duplicating elements here
      int64_t n = 1;
      unsigned inductions = arrayNode->getNumIndices();

      for (unsigned i = 0; i < inductions; i++) {
        const ast::bmodelica::ForIndex *ind = arrayNode->getIndex(i);
        auto indOperation =
            ind->getExpression()->cast<ast::bmodelica::Operation>();

        int64_t end;

        if (indOperation->getNumOfArguments() == 2) {
          end = indOperation->getArgument(1)
                    ->cast<ast::bmodelica::Constant>()
                    ->as<int64_t>();
        } else {
          end = indOperation->getArgument(2)
                    ->cast<ast::bmodelica::Constant>()
                    ->as<int64_t>();
        }

        n = n * end;
      }

      for (unsigned i = 0; i < n; i++) {
        s1.push_back(arrayNode->getValue());
      }
    } else {
      s2.push_back(node);
    }
  }

  while (!s2.empty()) {
    const ast::bmodelica::Expression *node = s2.pop_back_val();
    mlir::Location nodeLoc = loc(node->getLocation());
    auto loweredNode = lower(*node);
    if (!loweredNode) {
      return false;
    }
    outValues.push_back((*loweredNode)[0].get(nodeLoc));
  }

  return true;
}

std::optional<Results>
ArrayGeneratorLowerer::lower(const ast::bmodelica::ArrayConstant &array) {
  // TODO determine minimum required type
  mlir::Type elementType = RealType::get(builder().getContext());
  ;

  // Determine the shape.
  llvm::SmallVector<int64_t, 3> shape;
  computeShape(array, shape);

  // Determine the values.
  llvm::SmallVector<mlir::Value> values;
  if (!lowerValues(array, values)) {
    return std::nullopt;
  }

  auto tensorType = mlir::RankedTensorType::get(shape, elementType);
  mlir::Location location = loc(array.getLocation());

  mlir::Value result =
      builder().create<TensorFromElementsOp>(location, tensorType, values);

  return Reference::tensor(builder(), result);
}

std::optional<Results>
ArrayGeneratorLowerer::lower(const ast::bmodelica::ArrayForGenerator &array) {
  // TODO determine minimum required type
  mlir::Type elementType = RealType::get(builder().getContext());

  // Determine the shape.
  llvm::SmallVector<int64_t, 3> shape;
  computeShape(array, shape);

  mlir::Location location = loc(array.getLocation());

  const ast::bmodelica::Expression *topLevel = array.getValue();

  if (!topLevel->isa<ast::bmodelica::ArrayGenerator>()) {
    // Lower as a broadcast.
    mlir::Location nodeLoc = loc(topLevel->getLocation());
    auto loweredTopLevel = lower(*topLevel);
    if (!loweredTopLevel) {
      return std::nullopt;
    }
    mlir::Value elem = (*loweredTopLevel)[0].get(nodeLoc);

    mlir::Value result = builder().create<TensorBroadcastOp>(
        location, mlir::RankedTensorType::get(shape, elem.getType()), elem);

    return Reference::tensor(builder(), result);
  }

  // Flatten out all values.
  llvm::SmallVector<mlir::Value> values;
  if (!lowerValues(array, values)) {
    return std::nullopt;
  }

  auto tensorType = mlir::RankedTensorType::get(shape, elementType);

  mlir::Value result =
      builder().create<TensorFromElementsOp>(location, tensorType, values);

  return Reference::tensor(builder(), result);
}
} // namespace marco::codegen::lowering::bmodelica
