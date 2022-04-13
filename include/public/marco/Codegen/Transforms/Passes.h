#ifndef MARCO_CODEGEN_TRANSFORMS_PASSES_H
#define MARCO_CODEGEN_TRANSFORMS_PASSES_H

// Just a convenience header file to include the Modelica transformation passes

#include "marco/Codegen/Transforms/ModelSolving/ModelSolving.h"
#include "marco/Codegen/Transforms/ArrayDeallocation.h"
#include "marco/Codegen/Transforms/AutomaticDifferentiation.h"
#include "marco/Codegen/Transforms/ExplicitCastInsertion.h"
#include "marco/Codegen/Transforms/FunctionScalarization.h"
#include "marco/Codegen/Transforms/OpDistribution.h"

namespace marco::codegen
{
	inline void registerModelicaTransformationPasses()
	{
    registerArrayDeallocationPass();
		registerAutomaticDifferentiationPass();
		registerExplicitCastInsertionPass();
		registerFunctionScalarizationPass();
		registerSolveModelPass();

    // Debug transformations
    registerNegateOpDistributionPass();
    registerMulOpDistributionPass();
    registerDivOpDistributionPass();
	}
}

#endif //MARCO_CODEGEN_TRANSFORMS_PASSES_H
