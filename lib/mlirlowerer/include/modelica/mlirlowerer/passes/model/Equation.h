#pragma once

#include <llvm/Support/Error.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/utils/IndexSet.hpp>

#include "Expression.h"
#include "Path.h"

namespace modelica::codegen::model
{
	class AccessToVar;
	class Expression;
	class ExpressionPath;
	class VectorAccess;

	class Equation
	{
		public:
		using Ptr = std::shared_ptr<Equation>;

		Equation(mlir::Operation* op,
						 std::shared_ptr<Expression> left,
						 std::shared_ptr<Expression> right,
						 MultiDimInterval inductions = {},
						 bool isForward = true,
						 std::optional<EquationPath> path = std::nullopt);

		static Equation::Ptr build(EquationOp op);
		static Equation::Ptr build(ForEquationOp op);

		[[nodiscard]] mlir::Operation* getOp() const;

		[[nodiscard]] Expression& lhs();
		[[nodiscard]] const Expression& lhs() const;

		[[nodiscard]] Expression& rhs();
		[[nodiscard]] const Expression& rhs() const;

		[[nodiscard]] size_t amount() const;

		[[nodiscard]] const MultiDimInterval& getInductions() const;
		void setInductionVars(MultiDimInterval inductions);

		[[nodiscard]] bool isForEquation() const;
		[[nodiscard]] size_t dimensions() const;

		[[nodiscard]] bool isForward() const;
		void setForward(bool isForward);

		[[nodiscard]] bool isMatched() const;
		[[nodiscard]] Expression& getMatchedExp();
		[[nodiscard]] const Expression& getMatchedExp() const;
		void setMatchedExp(EquationPath path);

		[[nodiscard]] AccessToVar getDeterminedVariable() const;

		[[nodiscard]] ExpressionPath getMatchedExpressionPath() const;

		[[nodiscard]] Equation::Ptr normalized() const;

		[[nodiscard]] Equation::Ptr normalizeMatched() const;

		mlir::LogicalResult explicitate(mlir::OpBuilder& builder, size_t argumentIndex, bool left);
		mlir::LogicalResult explicitate(const ExpressionPath& path);
		mlir::LogicalResult explicitate();

		[[nodiscard]] Equation::Ptr clone() const;

		[[nodiscard]] Equation::Ptr composeAccess(const VectorAccess& transformation) const;

		template<typename Path>
		[[nodiscard]] Expression& reachExp(Path& path)
		{
			return path.isOnEquationLeftHand() ? path.reach(lhs()) : path.reach(rhs());
		}

		template<typename Path>
		[[nodiscard]] const Expression& reachExp(const Path& path) const
		{
			return path.isOnEquationLeftHand() ? path.reach(lhs()) : path.reach(rhs());
		}

		/**
		 * Tries to bring all the usages of the variable in the left-hand side
		 * of the equation to the left side of the equation.
		 */
		[[nodiscard]] Equation groupLeftHand() const;

		private:
		void getEquationsAmount(mlir::ValueRange values, llvm::SmallVectorImpl<long>& amounts) const;

		EquationSidesOp getTerminator();

		mlir::Operation* op;
		Expression::Ptr left;
		Expression::Ptr right;
		MultiDimInterval inductions;
		bool isForCycle;
		bool isForwardDirection;
		std::optional<EquationPath> matchedExpPath;
	};
}