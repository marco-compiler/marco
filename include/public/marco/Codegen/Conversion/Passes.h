#ifndef MARCO_CODEGEN_CONVERSION_PASSES_H
#define MARCO_CODEGEN_CONVERSION_PASSES_H

// Just a convenience header file to include the conversion passes

#include "marco/Codegen/Conversion/IDA/IDAToLLVM.h"
#include "marco/Codegen/Conversion/Modelica/LowerToCFG.h"
#include "marco/Codegen/Conversion/Modelica/LowerToLLVM.h"
#include "marco/Codegen/Conversion/Modelica/ModelicaConversion.h"

namespace marco::codegen
{
	inline void registerModelicaConversionPasses()
	{
		registerLowerToCFGPass();
		registerLLVMLoweringPass();
		registerFunctionConversionPass();
		registerIDAConversionPass();
		registerModelicaConversionPass();
		registerLowerToCFGPass();
	}
}

#endif // MARCO_CODEGEN_CONVERSION_PASSES_H
