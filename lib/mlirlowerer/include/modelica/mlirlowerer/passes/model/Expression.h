#pragma once

#include <mlir/IR/Operation.h>
#include <variant>

#include "Constant.h"
#include "Equation.h"
#include "Operation.h"
#include "Reference.h"

namespace modelica::codegen::model
{
	class Expression
	{
		public:
		Expression(mlir::Operation* op, Constant content);
		Expression(mlir::Operation* op, Reference content);
		Expression(mlir::Operation* op, Operation content);

		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<T>(content);
		}

		template<typename T>
		[[nodiscard]] T& get()
		{
			assert(isA<T>());
			return std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] const T& get() const
		{
			assert(isA<T>());
			return std::get<T>(content);
		}

		template<class Visitor>
		auto visit(Visitor&& vis)
		{
			return std::visit(std::forward<Visitor>(vis), content);
		}

		template<class Visitor>
		auto visit(Visitor&& vis) const
		{
			return std::visit(std::forward<Visitor>(vis), content);
		}

		[[nodiscard]] mlir::Operation* getOp();
		[[nodiscard]] mlir::Operation* getOp() const;

		[[nodiscard]] size_t childrenCount() const;

		[[nodiscard]] Expression& getChild(size_t index);
		[[nodiscard]] const Expression& getChild(size_t index) const;

		static Expression constant(mlir::Value value);
		static Expression reference(mlir::Value value);
		static Expression operation(mlir::Operation* operation, llvm::ArrayRef<Expression> args);

		template<typename... Args>
		static Expression operation(mlir::Operation* operation, Args&&... args)
		{
			return Expression::operation(operation, { args... });
		}

		private:
		mlir::Operation* op;
		std::variant<Constant, Reference, Operation> content;
	};

	class ExpressionPath
	{
		public:
		ExpressionPath(const Expression& exp, llvm::SmallVector<size_t, 3> path, bool left);
		ExpressionPath(const Expression& exp, EquationPath path);

		[[nodiscard]] EquationPath::const_iterator begin() const;
		[[nodiscard]] EquationPath::const_iterator end() const;

		[[nodiscard]] size_t depth() const;

		[[nodiscard]] const Expression& getExp() const;

		[[nodiscard]] const EquationPath& getEqPath() const;

		[[nodiscard]] bool isOnEquationLeftHand() const;

		[[nodiscard]] Expression& reach(Expression& exp) const;
		[[nodiscard]] const Expression& reach(const Expression& exp) const;

		private:
		EquationPath path;
		const Expression* exp;
	};
}