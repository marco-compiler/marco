#ifndef MARCO_CODEGEN_TRANSFORMS_PASSES_H
#define MARCO_CODEGEN_TRANSFORMS_PASSES_H

// Just a convenience header file to include the Modelica transformation passes

#include "marco/Codegen/Transforms/AutomaticDifferentiation.h"
#include "marco/Codegen/Transforms/BufferDeallocation.h"
#include "marco/Codegen/Transforms/ExplicitCastInsertion.h"
#include "marco/Codegen/Transforms/FunctionsVectorization.h"
#include "marco/Codegen/Transforms/ModelSolving.h"
#include "marco/Codegen/Transforms/OutputArraysPromotion.h"

namespace marco::codegen
{
	inline void registerModelicaTransformationPasses()
	{
		registerAutomaticDifferentiationPass();
		registerBufferDeallocationPass();
		registerExplicitCastInsertionPass();
		registerFunctionsVectorizationPass();
		registerOutputArraysPromotionPass();
		registerSolveModelPass();
	}
}

#endif //MARCO_CODEGEN_TRANSFORMS_PASSES_H
