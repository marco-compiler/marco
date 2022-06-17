#ifndef MARCO_CODEGEN_TRANSFORMS_PASSES_H
#define MARCO_CODEGEN_TRANSFORMS_PASSES_H

// Just a convenience header file to include the Modelica transformation passes

#include "marco/Codegen/Transforms/ModelSolving.h"
#include "marco/Codegen/Transforms/ArrayDeallocation.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation.h"
#include "marco/Codegen/Transforms/ExplicitCastInsertion.h"
#include "marco/Codegen/Transforms/FunctionScalarization.h"
#include "marco/Codegen/Transforms/Matching.h"
#include "marco/Codegen/Transforms/OpDistribution.h"
#include "marco/Codegen/Transforms/Scheduling.h"

namespace marco::codegen
{
  //===----------------------------------------------------------------------===//
  // Registration
  //===----------------------------------------------------------------------===//

  /// Generate the code for registering passes.
  #define GEN_PASS_REGISTRATION
  #include "marco/Codegen/Transforms/Passes.h.inc"
}

#endif // MARCO_CODEGEN_TRANSFORMS_PASSES_H
