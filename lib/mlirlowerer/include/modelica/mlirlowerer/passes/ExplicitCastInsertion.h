#pragma once

#include <mlir/Pass/Pass.h>

namespace modelica
{
	std::unique_ptr<mlir::Pass> createExplicitCastInsertionPass();
}
