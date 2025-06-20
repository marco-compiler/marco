#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_PASSES_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_PASSES_H

#include "marco/Dialect/BaseModelica/Transforms/AccessReplacementTest.h"
#include "marco/Dialect/BaseModelica/Transforms/AutomaticDifferentiation.h"
#include "marco/Dialect/BaseModelica/Transforms/BindingEquationConversion.h"
#include "marco/Dialect/BaseModelica/Transforms/CallCSE.h"
#include "marco/Dialect/BaseModelica/Transforms/DerivativeChainRule.h"
#include "marco/Dialect/BaseModelica/Transforms/DerivativesInitialization.h"
#include "marco/Dialect/BaseModelica/Transforms/DerivativesMaterialization.h"
#include "marco/Dialect/BaseModelica/Transforms/EquationAccessSplit.h"
#include "marco/Dialect/BaseModelica/Transforms/EquationExplicitation.h"
#include "marco/Dialect/BaseModelica/Transforms/EquationExplicitationTest.h"
#include "marco/Dialect/BaseModelica/Transforms/EquationFunctionLoopHoisting.h"
#include "marco/Dialect/BaseModelica/Transforms/EquationInductionsExplicitation.h"
#include "marco/Dialect/BaseModelica/Transforms/EquationSidesSplit.h"
#include "marco/Dialect/BaseModelica/Transforms/EquationTemplatesCreation.h"
#include "marco/Dialect/BaseModelica/Transforms/EulerForward.h"
#include "marco/Dialect/BaseModelica/Transforms/ExplicitCastInsertion.h"
#include "marco/Dialect/BaseModelica/Transforms/ExplicitInitialEquationsInsertion.h"
#include "marco/Dialect/BaseModelica/Transforms/ExplicitStartValueInsertion.h"
#include "marco/Dialect/BaseModelica/Transforms/FunctionDefaultValuesConversion.h"
#include "marco/Dialect/BaseModelica/Transforms/FunctionScalarization.h"
#include "marco/Dialect/BaseModelica/Transforms/FunctionUnwrap.h"
#include "marco/Dialect/BaseModelica/Transforms/IDA.h"
#include "marco/Dialect/BaseModelica/Transforms/InitialConditionsSolving.h"
#include "marco/Dialect/BaseModelica/Transforms/InliningAttributeInsertion.h"
#include "marco/Dialect/BaseModelica/Transforms/Matching.h"
#include "marco/Dialect/BaseModelica/Transforms/ModelAlgorithmConversion.h"
#include "marco/Dialect/BaseModelica/Transforms/ModelDebugCanonicalization.h"
#include "marco/Dialect/BaseModelica/Transforms/OpDistribution.h"
#include "marco/Dialect/BaseModelica/Transforms/PrintModelInfo.h"
#include "marco/Dialect/BaseModelica/Transforms/PureFunctionInlining.h"
#include "marco/Dialect/BaseModelica/Transforms/RangeBoundariesInference.h"
#include "marco/Dialect/BaseModelica/Transforms/ReadOnlyVariablesPropagation.h"
#include "marco/Dialect/BaseModelica/Transforms/RecordInlining.h"
#include "marco/Dialect/BaseModelica/Transforms/RungeKutta.h"
#include "marco/Dialect/BaseModelica/Transforms/SCCAbsenceVerification.h"
#include "marco/Dialect/BaseModelica/Transforms/SCCDetection.h"
#include "marco/Dialect/BaseModelica/Transforms/SCCSolvingBySubstitution.h"
#include "marco/Dialect/BaseModelica/Transforms/SCCSolvingWithKINSOL.h"
#include "marco/Dialect/BaseModelica/Transforms/ScalarRangesEquationSplit.h"
#include "marco/Dialect/BaseModelica/Transforms/ScheduleParallelization.h"
#include "marco/Dialect/BaseModelica/Transforms/SchedulersInstantiation.h"
#include "marco/Dialect/BaseModelica/Transforms/Scheduling.h"
#include "marco/Dialect/BaseModelica/Transforms/SingleValuedInductionElimination.h"
#include "marco/Dialect/BaseModelica/Transforms/VariablesPromotion.h"
#include "marco/Dialect/BaseModelica/Transforms/VariablesPruning.h"
#include "marco/Dialect/BaseModelica/Transforms/ViewAccessFolding.h"

namespace mlir::bmodelica {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_PASSES_H
