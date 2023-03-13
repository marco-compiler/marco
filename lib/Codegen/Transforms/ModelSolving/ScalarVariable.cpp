#include "marco/Codegen/Transforms/ModelSolving/ScalarVariable.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  ScalarVariable::ScalarVariable(VariableOp variableOp)
    : BaseVariable(variableOp)
  {
    assert(variableOp.getVariableType().isScalar());
  }

  std::unique_ptr<Variable> ScalarVariable::clone() const
  {
    return std::make_unique<ScalarVariable>(*this);
  }

  size_t ScalarVariable::getRank() const
  {
    return 1;
  }

  long ScalarVariable::getDimensionSize(size_t index) const
  {
    return 1;
  }

  IndexSet ScalarVariable::getIndices() const
  {
    return IndexSet(MultidimensionalRange(Range(0, 1)));
  }
}