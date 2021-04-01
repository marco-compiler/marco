#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica::codegen
{
	std::unique_ptr<mlir::Pass> createExplicitCastInsertionPass();
}
