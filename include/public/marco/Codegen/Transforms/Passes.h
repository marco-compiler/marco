#ifndef MARCO_CODEGEN_TRANSFORMS_PASSES_H
#define MARCO_CODEGEN_TRANSFORMS_PASSES_H

#include "marco/Codegen/Transforms/AccessReplacementTest.h"
#include "marco/Codegen/Transforms/ArrayDeallocation.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation.h"
#include "marco/Codegen/Transforms/BindingEquationConversion.h"
#include "marco/Codegen/Transforms/CyclesSolving.h"
#include "marco/Codegen/Transforms/DerivativesAllocation.h"
#include "marco/Codegen/Transforms/EquationAccessSplit.h"
#include "marco/Codegen/Transforms/EquationExplicitationTest.h"
#include "marco/Codegen/Transforms/EquationTemplatesCreation.h"
#include "marco/Codegen/Transforms/EquationViewsComputation.h"
#include "marco/Codegen/Transforms/EulerForward.h"
#include "marco/Codegen/Transforms/ExplicitCastInsertion.h"
#include "marco/Codegen/Transforms/ExplicitInitialEquationsInsertion.h"
#include "marco/Codegen/Transforms/ExplicitStartValueInsertion.h"
#include "marco/Codegen/Transforms/FunctionInlining.h"
#include "marco/Codegen/Transforms/FunctionScalarization.h"
#include "marco/Codegen/Transforms/IDA.h"
#include "marco/Codegen/Transforms/Matching.h"
#include "marco/Codegen/Transforms/ModelAlgorithmConversion.h"
#include "marco/Codegen/Transforms/OpDistribution.h"
#include "marco/Codegen/Transforms/RangeBoundariesInference.h"
#include "marco/Codegen/Transforms/ReadOnlyVariablesPropagation.h"
#include "marco/Codegen/Transforms/RecordInlining.h"
#include "marco/Codegen/Transforms/Scheduling.h"
#include "marco/Codegen/Transforms/SolverPassBase.h"
#include "marco/Codegen/Transforms/VariablesPromotion.h"

namespace marco::codegen
{
  /// Generate the code for registering passes.
  #define GEN_PASS_REGISTRATION
  #include "marco/Codegen/Transforms/Passes.h.inc"
}

#endif // MARCO_CODEGEN_TRANSFORMS_PASSES_H
