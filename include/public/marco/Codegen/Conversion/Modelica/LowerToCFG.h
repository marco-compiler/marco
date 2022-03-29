#ifndef MARCO_CODEN_CONVERSION_MODELICA_LOWERTOCFG_H
#define MARCO_CODEN_CONVERSION_MODELICA_LOWERTOCFG_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	/**
	 * Convert the control flow operations of the Modelica and the SCF
	 * dialects.
	 *
	 * @param bitWidth bit width
	 */
	std::unique_ptr<mlir::Pass> createLowerToCFGPass(unsigned int bitWidth = 64);

	inline void registerLowerToCFGPass()
	{
		mlir::registerPass("convert-modelica-to-cfg", "Modelica: convert to CFG",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createLowerToCFGPass();
											 });
	}
}

#endif // MARCO_CODEN_CONVERSION_MODELICA_LOWERTOCFG_H
