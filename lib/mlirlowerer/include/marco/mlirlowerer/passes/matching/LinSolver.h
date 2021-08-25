#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/Support/LogicalResult.h>
#include <marco/mlirlowerer/passes/model/Model.h>

namespace marco::codegen::model
{
	class Equation;

	/**
	 * Get the expression that represents the variables explicitated by source
	 * and replace each occurrence of that variable inside destination with the
	 * aforementioned expression.
	 *
	 * @param builder operation builder
	 * @param source equation containing the explicitated variable to be replaced
	 * @param destination equation inside which the source variable occurrences have to be replaced
	 */
	void replaceUses(mlir::OpBuilder& builder, const Equation& source, Equation& destination);

	/**
	 * Solve the linear system by using the variable elimination method.
	 * Starting from the last equation eq_n (and its explicitated member x_n),
	 * we replace in the all other equations the declarations of x_n with the
	 * right-hand side of eq_n, which by design doesn't contain any other use
	 * of x_n. Then proceed with the second-last equation and so on.
	 *
	 * @param builder operation builder
	 * @param equations system of equations
	 * @return success if everything went right
	 */
	mlir::LogicalResult linearySolve(mlir::OpBuilder& builder, llvm::SmallVectorImpl<Equation>& equations);

	/**
	 * Check if the given system of equations can be solved by the currently
	 * implemented algorithm. The current algorithm cannot solve systems where
	 * the algebraic loop is composed by variables in the same array or when there
	 * is a derivative operation.
	 *
	 * @param equations system of equations
	 * @return true if the system of equations can be solved
	 */
	bool canSolveSystem(llvm::SmallVectorImpl<Equation>& equations, const Model& model);
}	 // namespace marco::codegen::model
