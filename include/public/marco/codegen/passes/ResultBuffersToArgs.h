#pragma once

#include "marco/codegen/dialects/modelica/Ops.h"
#include "marco/codegen/passes/TypeConverter.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace marco::codegen
{
	std::unique_ptr<mlir::Pass> createResultBuffersToArgsPass();

	inline void registerResultBuffersToArgsPass()
	{
		mlir::registerPass("result-buffers-to-args", "Modelica: result buffers to args",
											 []() -> std::unique_ptr<::mlir::Pass> {
												 return createResultBuffersToArgsPass();
											 });
	}
}
