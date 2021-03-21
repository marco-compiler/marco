#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica
{
	/**
	 * Create a pass to convert Modelica operations to a mix of Std,
	 * SCF and LLVM ones.
 	 */
	std::unique_ptr<mlir::Pass> createModelicaConversionPass();
}
