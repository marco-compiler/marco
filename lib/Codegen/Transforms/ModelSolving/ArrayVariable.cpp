#include "marco/Codegen/Transforms/ModelSolving/ArrayVariable.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  ArrayVariable::ArrayVariable(mlir::Value value)
    : BaseVariable(value)
  {
    assert(!value.getType().cast<ArrayType>().isScalar());
  }

  std::unique_ptr<Variable> ArrayVariable::clone() const
  {
    return std::make_unique<ArrayVariable>(*this);
  }

  size_t ArrayVariable::getRank() const
  {
    auto result = getValue().getType().cast<ArrayType>().getRank();
    assert(result != 0);
    return result;
  }

  long ArrayVariable::getDimensionSize(size_t index) const
  {
    return getValue().getType().cast<ArrayType>().getShape()[index];
  }

  IndexSet ArrayVariable::getIndices() const
  {
    std::vector<Range> ranges;

    for (size_t i = 0, e = getRank(); i < e; ++i) {
      ranges.emplace_back(0, getDimensionSize(i));
    }

    return IndexSet(MultidimensionalRange(std::move(ranges)));
  }
}