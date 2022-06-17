#ifndef MARCO_CODEGEN_TRANSFORMS_PASSDETAIL_H
#define MARCO_CODEGEN_TRANSFORMS_PASSDETAIL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir
{
  namespace ida {
    class IDADialect;
  }

  namespace modelica {
    class ModelicaDialect;
  }

  namespace arith {
    class ArithmeticDialect;
  }

  namespace cf {
    class ControlFlowDialect;
  }

  namespace func {
    class FuncDialect;
  }

  namespace LLVM {
    class LLVMDialect;
  }

  namespace scf {
    class SCFDialect;
  }
}

namespace marco::codegen
{
#define GEN_PASS_CLASSES
#include "marco/Codegen/Transforms/Passes.h.inc"
}

#endif // MARCO_CODEGEN_TRANSFORMS_PASSDETAIL_H
