#pragma once

// Just a convenience header file to include the Modelica passes

#include "passes/AutomaticDifferentiation.h"
#include "passes/BufferDeallocation.h"
#include "passes/ExplicitCastInsertion.h"
#include "passes/FunctionsVectorization.h"
#include "passes/LowerToCFG.h"
#include "passes/LowerToLLVM.h"
#include "passes/ModelicaConversion.h"
#include "passes/ResultBuffersToArgs.h"
#include "passes/SolveModel.h"

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
