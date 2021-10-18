#pragma once

#include <mlir/Pass/Pass.h>

namespace marco::codegen
{
	std::unique_ptr<mlir::Pass> createBufferDeallocationPass();

	inline void registerBufferDeallocationPass()
	{
		mlir::registerPass("modelica-buffer-deallocation", "Modelica: automatic buffer deallocation",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createBufferDeallocationPass();
											 });
	}
}
