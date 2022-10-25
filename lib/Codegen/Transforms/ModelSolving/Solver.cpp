#include "marco/Codegen/Transforms/ModelSolving/Solver.h"
#include "llvm/ADT/StringSwitch.h"

using namespace marco;
using namespace marco::codegen;

static Solver::Kind getKindFromName(llvm::StringRef solverName)
{
  return llvm::StringSwitch<Solver::Kind>(solverName)
      .Case("forward-euler", Solver::Kind::forwardEuler)
      .Case("ida", Solver::Kind::ida)
      .Default(Solver::Kind::forwardEuler);
}

namespace marco::codegen
{
  Solver::Solver() : Solver("")
  {
  }

  Solver::Solver(const char* solverName)
    : kind(getKindFromName(solverName))
  {
  }

  Solver::Solver(Kind kind) : kind(kind)
  {
  }

  Solver::Kind Solver::getKind() const
  {
    return kind;
  }

  Solver Solver::forwardEuler()
  {
    return Solver(Kind::forwardEuler);
  }

  Solver Solver::ida()
  {
    return Solver(Kind::ida);
  }
}
