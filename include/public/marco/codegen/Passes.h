#pragma once

// Just a convenience header file to include the Modelica passes

#include "marco/codegen/passes/AutomaticDifferentiation.h"
#include "marco/codegen/passes/BufferDeallocation.h"
#include "marco/codegen/passes/ExplicitCastInsertion.h"
#include "marco/codegen/passes/FunctionsVectorization.h"
#include "marco/codegen/passes/LowerToCFG.h"
#include "marco/codegen/passes/LowerToLLVM.h"
#include "marco/codegen/passes/ModelicaConversion.h"
#include "marco/codegen/passes/ResultBuffersToArgs.h"
#include "marco/codegen/passes/SolveModel.h"

namespace marco::codegen
{
	inline void registerModelicaPasses()
	{
		registerAutomaticDifferentiationPass();
		registerBufferDeallocationPass();
		registerExplicitCastInsertionPass();
		registerFunctionsVectorizationPass();
		registerLowerToCFGPass();
		registerLLVMLoweringPass();
		registerFunctionConversionPass();
		registerModelicaConversionPass();
		registerLowerToCFGPass();
		registerResultBuffersToArgsPass();
		registerSolveModelPass();
	}
}
