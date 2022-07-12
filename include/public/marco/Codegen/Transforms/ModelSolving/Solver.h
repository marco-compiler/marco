#ifndef MARCO_CODEGEN_TRANSFORMS_SOLVER_H
#define MARCO_CODEGEN_TRANSFORMS_SOLVER_H

#include "llvm/ADT/StringRef.h"
#include <string>

namespace marco::codegen
{
  class Solver
  {
    public:
      enum class Kind
      {
        forwardEuler,
        ida,
        kinsol
      };

      Solver();

      Solver(const char* solverName);

      Kind getKind() const;

      static Solver forwardEuler();
      static Solver ida();
      static Solver kinsol();

    private:
      Solver(Kind kind);

    private:
      Kind kind;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_SOLVER_H
