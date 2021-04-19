#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica::codegen
{
	/**
	 * Create a pass to convert Modelica operations to a mix of Std,
	 * SCF and LLVM ones.
 	 */
	std::unique_ptr<mlir::Pass> createModelicaConversionPass();

	/**
	 * Convert the control flow operations of the Modelica and the SCF
	 * dialects.
	 */
	std::unique_ptr<mlir::Pass> createLowerToCFGPass();
}
