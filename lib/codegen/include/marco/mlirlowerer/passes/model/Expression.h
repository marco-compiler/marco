#pragma once

#include <llvm/Support/Casting.h>
#include <memory>
#include <mlir/IR/Operation.h>
#include <variant>

#include "Constant.h"
#include "Operation.h"
#include "Reference.h"

namespace marco::codegen::model
{
	class Constant;
	class Reference;
	class Operation;

	class Expression
	{
		private:
		class Impl
		{
			public:
			Impl(mlir::Operation* op, Constant content);
			Impl(mlir::Operation* op, Reference content);
			Impl(mlir::Operation* op, Operation content);

			private:
			friend class Expression;

			mlir::Operation* op;
			std::variant<Constant, Reference, Operation> content;
			std::string name;
		};

		std::shared_ptr<Impl> impl;

		public:
		Expression(mlir::Operation* op, Constant content);
		Expression(mlir::Operation* op, Reference content);
		Expression(mlir::Operation* op, Operation content);

		bool operator==(const Expression& rhs) const;
		bool operator!=(const Expression& rhs) const;

		static Expression build(mlir::Value value);

		static Expression constant(mlir::Value value);
		static Expression reference(mlir::Value value);
		static Expression operation(mlir::Operation* operation, llvm::ArrayRef<Expression> args);

		template<typename... Args>
		static Expression operation(mlir::Operation* operation, Args&&... args)
		{
			return Expression::operation(operation, { args... });
		}

		template<typename T>
		[[nodiscard]] T& get()
		{
			assert(std::holds_alternative<T>(impl->content));
			return std::get<T>(impl->content);
		}

		template<typename T>
		[[nodiscard]] const T& get() const
		{
			assert(std::holds_alternative<T>(impl->content));
			return std::get<T>(impl->content);
		}

		template<class Visitor>
		auto visit(Visitor&& vis)
		{
			return std::visit(std::forward<Visitor>(vis), impl->content);
		}

		template<class Visitor>
		auto visit(Visitor&& vis) const
		{
			return std::visit(std::forward<Visitor>(vis), impl->content);
		}

		[[nodiscard]] mlir::Operation* getOp() const;

		[[nodiscard]] bool isConstant() const;
		[[nodiscard]] bool isReference() const;
		[[nodiscard]] bool isReferenceAccess() const;
		[[nodiscard]] bool isOperation() const;

		[[nodiscard]] size_t childrenCount() const;

		[[nodiscard]] Expression getChild(size_t index) const;

		[[nodiscard]] mlir::Value getReferredVectorAccess() const;
		[[nodiscard]] Expression& getReferredVectorAccessExp();
		[[nodiscard]] const Expression& getReferredVectorAccessExp() const;
	};
}