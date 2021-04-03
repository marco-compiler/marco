#pragma once

#include <llvm/Support/Error.h>
#include <mlir/IR/Operation.h>
#include <modelica/utils/IndexSet.hpp>

namespace modelica::codegen::model
{
	class AccessToVar;
	class Expression;
	class ExpressionPath;
	class VectorAccess;

	class EquationTemplate
	{
		public:
		EquationTemplate(Expression left, Expression right, std::string name);

		[[nodiscard]] Expression& lhs();
		[[nodiscard]] const Expression& lhs() const;

		[[nodiscard]] Expression& rhs();
		[[nodiscard]] const Expression& rhs() const;

		[[nodiscard]] std::string& getName();
		[[nodiscard]] const std::string& getName() const;
		void setName(std::string newName);

		void swapLeftRight();

		private:
		std::shared_ptr<Expression> left;
		std::shared_ptr<Expression> right;
		std::string name;
	};

	class EquationPath
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T>;

		public:
		using iterator = Container<size_t>::iterator;
		using const_iterator = Container<size_t>::const_iterator;

		EquationPath(llvm::SmallVector<size_t, 3> path, bool left);

		[[nodiscard]] const_iterator begin() const;
		[[nodiscard]] const_iterator end() const;

		[[nodiscard]] size_t depth() const;
		[[nodiscard]] bool isOnEquationLeftHand() const;

		[[nodiscard]] Expression& reach(Expression& exp) const;
		[[nodiscard]] const Expression& reach(const Expression& exp) const;

		private:
		Container<size_t> path;
		bool left;
	};

	class Equation
	{
		public:
		Equation(mlir::Operation* op,
						 Expression left,
						 Expression right,
						 std::string templateName = "",
						 MultiDimInterval inductions = {},
						 bool isForward = true,
						 std::optional<EquationPath> path = std::nullopt);

		Equation(mlir::Operation* op,
						 std::shared_ptr<EquationTemplate> templ,
						 MultiDimInterval interval,
						 bool isForward);

		[[nodiscard]] mlir::Operation* getOp() const;

		[[nodiscard]] Expression& lhs();
		[[nodiscard]] const Expression& lhs() const;

		[[nodiscard]] Expression& rhs();
		[[nodiscard]] const Expression& rhs() const;

		[[nodiscard]] size_t amount() const;

		[[nodiscard]] std::shared_ptr<EquationTemplate>& getTemplate();
		[[nodiscard]] const std::shared_ptr<EquationTemplate>& getTemplate() const;

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


		[[nodiscard]] Equation normalized() const;

		[[nodiscard]] Equation normalizeMatched() const;

		mlir::LogicalResult explicitate(size_t argumentIndex, bool left);
		mlir::LogicalResult explicitate(const ExpressionPath& path);
		mlir::LogicalResult explicitate();

		[[nodiscard]] Equation clone(std::string newName) const;

		[[nodiscard]] Equation composeAccess(const VectorAccess& transformation) const;

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

		private:
		void getEquationsAmount(mlir::ValueRange values, llvm::SmallVectorImpl<long>& amounts) const;

		mlir::Operation* op;
		std::shared_ptr<EquationTemplate> body;
		MultiDimInterval inductions;
		bool isForCycle;
		bool isForwardDirection;
		std::optional<EquationPath> matchedExpPath;
	};
}