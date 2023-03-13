#include "marco/Codegen/Transforms/ModelSolving/ArrayVariable.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  ArrayVariable::ArrayVariable(VariableOp memberCreateOp)
    : BaseVariable(memberCreateOp)
  {
    assert(!memberCreateOp.getMemberType().isScalar());
  }

  std::unique_ptr<Variable> ArrayVariable::clone() const
  {
    return std::make_unique<ArrayVariable>(*this);
  }

  size_t ArrayVariable::getRank() const
  {
    int64_t rank = getDefiningOp().getMemberType().getRank();
    assert(rank != 0);
    return rank;
  }

  long ArrayVariable::getDimensionSize(size_t index) const
  {
    return getDefiningOp().getMemberType().getShape()[index];
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