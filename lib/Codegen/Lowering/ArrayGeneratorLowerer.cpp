#include "marco/Codegen/Lowering/ArrayGeneratorLowerer.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ArrayGeneratorLowerer::ArrayGeneratorLowerer(BridgeInterface* bridge)
      : Lowerer(bridge)
  {
  }

  Results ArrayGeneratorLowerer::lower(const ast::ArrayGenerator& array)
  {
    if (array.isa<ast::ArrayConstant>()) {
      return lower(*array.cast<ast::ArrayConstant>());
    } else if (array.isa<ast::ArrayForGenerator>()) {
      return lower(*array.cast<ast::ArrayForGenerator>());
    } else {
      llvm_unreachable("Unknown type of array generator");
    }
  }

  void ArrayGeneratorLowerer::computeShape(const ast::ArrayGenerator& array, llvm::SmallVectorImpl<int64_t>& outShape)
  {
    llvm::SmallVector<const ast::ArrayGenerator*> nestedArrays;
    nestedArrays.push_back(&array);

    while (!nestedArrays.empty()) {
      const ast::ArrayGenerator* current = nestedArrays.pop_back_val();

      if (auto constant = current->dyn_cast<ast::ArrayConstant>()) {
        size_t numOfChildren = constant->size();
        outShape.push_back(numOfChildren);

        if (numOfChildren > 0) {
          // TODO: check if the various children have a consistent type and size
          const ast::Expression* child = (*constant)[0];

          if (auto arrayChild = child->dyn_cast<ast::ArrayGenerator>()) {
            nestedArrays.push_back(arrayChild);
          }
        }

      } else if (auto forGen = current->dyn_cast<ast::ArrayForGenerator>()) {
        unsigned inductions = forGen->getNumIndices();
        for (unsigned i = 0; i < inductions; i++) {
          const ast::Induction *ind = forGen->getIndex(inductions - i - 1);
          
          if (!ind->getBegin()->isa<ast::Constant>() ||
              !ind->getEnd()->isa<ast::Constant>() ||
              !ind->getStep()->isa<ast::Constant>()) {
            assert(false && "Array for generators with non-constant indices not supported");
          }
          auto begin = ind->getBegin()->cast<ast::Constant>()->as<uint64_t>();
          auto end = ind->getEnd()->cast<ast::Constant>()->as<uint64_t>();
          auto step = ind->getStep()->cast<ast::Constant>()->as<uint64_t>();
          if (begin != 1 || step != 1) {
            assert(false && "Array for generators with step/index not equal to 1 are not supported");
          }

          outShape.push_back(end);
        }

        if (auto arrayChild = forGen->getValue()->dyn_cast<ast::ArrayGenerator>()) {
          nestedArrays.push_back(arrayChild);
        }
      }
    }
  }

  void ArrayGeneratorLowerer::lowerValues(const ast::Expression& array, llvm::SmallVectorImpl<mlir::Value>& outValues)
  {
    llvm::SmallVector<const ast::Expression*> s1;
    llvm::SmallVector<const ast::Expression*> s2;

    s1.push_back(&array);

    while (!s1.empty()) {
      const ast::Expression* node = s1.pop_back_val();

      if (auto arrayNode = node->dyn_cast<ast::ArrayConstant>()) {
        for (size_t i = 0, e = arrayNode->size(); i < e; ++i) {
          s1.push_back((*arrayNode)[i]);
        }
      } else if (auto arrayNode = node->dyn_cast<ast::ArrayForGenerator>()) {
        // TODO: extend broadcast to arrays and avoid duplicating elements here
        unsigned n = 1;
        unsigned inductions = arrayNode->getNumIndices();
        for (unsigned i = 0; i < inductions; i++) {
          const ast::Induction *ind = arrayNode->getIndex(i);          
          auto end = ind->getEnd()->cast<ast::Constant>()->as<uint64_t>();
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
      const ast::Expression* node = s2.pop_back_val();
      mlir::Location nodeLoc = loc(node->getLocation());
      outValues.push_back(lower(*node)[0].get(nodeLoc));
    }
  }

  Results ArrayGeneratorLowerer::lower(const ast::ArrayConstant& array)
  {
    // TODO determine minimum required type
    mlir::Type elementType = RealType::get(builder().getContext());;

    // Determine the shape.
    llvm::SmallVector<int64_t, 3> shape;
    computeShape(array, shape);

    // Determine the values.
    llvm::SmallVector<mlir::Value> values;
    lowerValues(array, values);

    auto arrayType = ArrayType::get(shape, elementType);
    mlir::Location location = loc(array.getLocation());
    mlir::Value result = builder().create<ArrayFromElementsOp>(
        location, arrayType, values);

    return Reference::memory(builder(), result);
  }

  Results ArrayGeneratorLowerer::lower(const ast::ArrayForGenerator& array)
  {
    // TODO determine minimum required type
    mlir::Type elementType = RealType::get(builder().getContext());

    // Determine the shape.
    llvm::SmallVector<int64_t, 3> shape;
    computeShape(array, shape);

    auto arrayType = ArrayType::get(shape, elementType);
    mlir::Location location = loc(array.getLocation());

    const ast::Expression *topLevel = array.getValue();
    if (!topLevel->isa<ast::ArrayGenerator>()) {
      // Lower as a broadcast
      mlir::Location nodeLoc = loc(topLevel->getLocation());
      mlir::Value elem = lower(*topLevel)[0].get(nodeLoc);
      mlir::Value result = builder().create<ArrayBroadcastOp>(location, arrayType, elem);
      return Reference::memory(builder(), result);
    }

    // Flatten out all values
    llvm::SmallVector<mlir::Value> values;
    lowerValues(array, values);
    mlir::Value result = builder().create<ArrayFromElementsOp>(
        location, arrayType, values);
    return Reference::memory(builder(), result);
  }
}
