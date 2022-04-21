#include "marco/Codegen/Transforms/ModelSolving/Model.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  Variables discoverVariables(ModelOp model)
  {
    Variables result;

    for (const auto& var : model.bodyRegion().getArguments()) {
      result.add(std::make_unique<Variable>(var));
    }

    return result;
  }

  Equations<Equation> discoverEquations(ModelOp model, const Variables& variables)
  {
    Equations<Equation> result;

    model.walk([&](EquationOp equationOp) {
      result.add(Equation::build(equationOp, variables));
    });

    return result;
  }
}
