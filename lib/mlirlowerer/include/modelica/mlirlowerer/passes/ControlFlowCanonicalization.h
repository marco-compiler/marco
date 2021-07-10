#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica::codegen
{
	std::unique_ptr<mlir::Pass> createControlFlowCanonicalizationPass(unsigned int bitWidth = 64);

	inline void registerControlFlowCanonicalizationPass()
	{
		mlir::registerPass("canonicalize-cfg", "Modelica: canonicalize CFG operations",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createControlFlowCanonicalizationPass();
											 });
	}
}
