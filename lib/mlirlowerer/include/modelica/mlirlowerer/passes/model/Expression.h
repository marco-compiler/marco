#pragma once

#include <llvm/Support/Casting.h>
#include <memory>
#include <mlir/IR/Operation.h>
#include <variant>

#include "Constant.h"
#include "Operation.h"
#include "Reference.h"

namespace modelica::codegen::model
{
	class Constant;
	class Reference;
	class Operation;

	class Expression
	{
		public:
		using Ptr = std::shared_ptr<Expression>;

		Expression(mlir::Operation* op, Constant content);
		Expression(mlir::Operation* op, Reference content);
		Expression(mlir::Operation* op, Operation content);

		static Expression::Ptr build(mlir::Value value);

		static Expression::Ptr constant(mlir::Value value);
		static Expression::Ptr reference(mlir::Value value);
		static Expression::Ptr operation(mlir::Operation* operation, llvm::ArrayRef<Expression::Ptr> args);

		template<typename... Args>
		static Expression::Ptr operation(mlir::Operation* operation, Args&&... args)
		{
			return Expression::operation(operation, { args... });
		}

		template<typename T>
		[[nodiscard]] T& get()
		{
			assert(std::holds_alternative<T>(content));
			return std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] const T& get() const
		{
			assert(std::holds_alternative<T>(content));
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

		[[nodiscard]] mlir::Operation* getOp() const;

		[[nodiscard]] bool isConstant() const;
		[[nodiscard]] bool isReference() const;
		[[nodiscard]] bool isReferenceAccess() const;
		[[nodiscard]] bool isOperation() const;

		[[nodiscard]] size_t childrenCount() const;

		[[nodiscard]] Expression::Ptr getChild(size_t index) const;

		[[nodiscard]] mlir::Value getReferredVectorAccess() const;
		[[nodiscard]] Expression& getReferredVectorAccessExp();
		[[nodiscard]] const Expression& getReferredVectorAccessExp() const;

		private:
		mlir::Operation* op;
		std::variant<Constant, Reference, Operation> content;
		std::string name;
	};
}