#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/Support/LogicalResult.h>

namespace modelica::codegen::model
{
	class Equation;

	/**
	 * Solve the linear system by using the variable elimination method.
	 * Starting from the last equation eq_n (and its explicitated member x_n),
	 * we replace in the all other equations the declarations of x_n with the
	 * right-hand side of eq_n, which by design doesn't contain any other use
	 * of x_n. Then proceed with the second-last equation and so on.
	 *
	 * @param builder		operation builder
	 * @param equations system of equations
	 * @return success if everything went right
	 */
	mlir::LogicalResult linearySolve(mlir::OpBuilder& builder, llvm::SmallVectorImpl<Equation>& equations);
}
