#ifndef MARCO_CODEGEN_CONVERSION_PASSES_H
#define MARCO_CODEGEN_CONVERSION_PASSES_H

// Just a convenience header file to include the conversion passes

#include "marco/Codegen/Conversion/Modelica/LowerToCFG.h"
#include "marco/Codegen/Conversion/Modelica/LowerToLLVM.h"
#include "marco/Codegen/Conversion/Modelica/ModelicaConversion.h"
#include "marco/Codegen/Conversion/IDA/IDAToLLVM.h"
#include "marco/Codegen/Conversion/Generic/UnrealizedCastReconciliation.h"

namespace marco::codegen
{
	inline void registerModelicaConversionPasses()
	{
		registerLowerToCFGPass();
		registerLLVMLoweringPass();
		registerIDAConversionPass();
		registerModelicaConversionPass();
		registerLowerToCFGPass();
	}

  inline void registerIDAConversionPasses()
  {
    registerIDAConversionPass();
  }
}

#endif // MARCO_CODEGEN_CONVERSION_PASSES_H
