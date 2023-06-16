#include "marco/Codegen/Lowering/ArrayGeneratorLowerer.h"
#include <stack>

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

  Results ArrayGeneratorLowerer::lower(const ast::ArrayConstant& array)
  {
    mlir::Location location = loc(array.getLocation());

    llvm::SmallVector<int64_t, 3> shape;
    llvm::SmallVector<mlir::Value> values;

    // TODO determine minimum required type
    mlir::Type elementType = RealType::get(builder().getContext());;

    std::stack<const ast::ArrayConstant*> nestedArrays;

    // Determine the shape.
    nestedArrays.push(&array);

    while (!nestedArrays.empty()) {
      const ast::ArrayConstant* current = nestedArrays.top();
      nestedArrays.pop();

      size_t numOfChildren = current->size();
      shape.push_back(numOfChildren);

      if (numOfChildren > 0) {
        const ast::Expression* child = (*current)[0];

        if (auto arrayChild = child->dyn_cast<ast::ArrayConstant>()) {
          nestedArrays.push(arrayChild);
        }
      }
    }

    // Determine the values.
    std::stack<const ast::Expression*> s1;
    std::stack<const ast::Expression*> s2;

    s1.push(&array);

    while (!s1.empty()) {
      const ast::Expression* node = s1.top();
      s1.pop();

      if (auto arrayNode = node->dyn_cast<ast::ArrayConstant>()) {
        for (size_t i = 0, e = arrayNode->size(); i < e; ++i) {
          s1.push((*arrayNode)[i]);
        }
      } else {
        s2.push(node);
      }
    }

    while (!s2.empty()) {
      const ast::Expression* node = s2.top();
      s2.pop();

      mlir::Location nodeLoc = loc(node->getLocation());
      values.push_back(lower(*node)[0].get(nodeLoc));
    }

    auto arrayType = ArrayType::get(shape, elementType);

    mlir::Value result = builder().create<ArrayFromElementsOp>(
        location, arrayType, values);

    return Reference::memory(builder(), result);
  }

  Results ArrayGeneratorLowerer::lower(const ast::ArrayForGenerator& array)
  {
    llvm_unreachable("Unsupported kind of array generator");
  }
}
