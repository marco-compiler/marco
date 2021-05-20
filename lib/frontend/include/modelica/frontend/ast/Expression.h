#pragma once

#include <initializer_list>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <utility>
#include <variant>

#include "Array.h"
#include "ASTNode.h"
#include "Call.h"
#include "Constant.h"
#include "Operation.h"
#include "ReferenceAccess.h"
#include "Tuple.h"
#include "Type.h"

namespace modelica::frontend
{
	class Expression
			: public impl::Cloneable<Expression>,
				public impl::Dumpable<Expression>
	{
		public:
		Expression(const Expression& other);
		Expression(Expression&& other);

		~Expression();

		Expression& operator=(const Expression& other);
		Expression& operator=(Expression&& other);

		friend void swap(Expression& first, Expression& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool operator==(const Expression& other) const;
		[[nodiscard]] bool operator!=(const Expression& other) const;

		template<typename T>
		[[nodiscard]] bool isa() const
		{
			return std::holds_alternative<T>(content);
		}

		template<typename T>
		[[nodiscard]] T* get()
		{
			assert(isa<T>());
			return &std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] const T* get() const
		{
			assert(isa<T>());
			return &std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] T* dyn_get()
		{
			if (!isa<T>())
				return nullptr;

			return get<T>();
		}

		template<typename T>
		[[nodiscard]] const T* dyn_get() const
		{
			if (!isa<T>())
				return nullptr;

			return get<T>();
		}

		template<typename Visitor>
		auto visit(Visitor&& visitor)
		{
			return std::visit(visitor, content);
		}

		template<typename Visitor>
		auto visit(Visitor&& visitor) const
		{
			return std::visit(visitor, content);
		}

		[[nodiscard]] SourceRange getLocation() const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type tp);

		[[nodiscard]] bool isLValue() const;

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Expression> array(SourceRange location, Type type, Args&&... args)
		{
			Array content(std::move(location), std::move(type), std::forward<Args>(args)...);
			return std::unique_ptr<Expression>(new Expression(std::move(content)));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Expression> call(SourceRange location, Type type, std::unique_ptr<Expression> function, Args&&... args)
		{
			Call content(std::move(location), std::move(type), std::move(function), std::forward<Args>(args)...);
			return std::unique_ptr<Expression>(new Expression(std::move(content)));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Expression> constant(SourceRange location, Type type, Args&&... args)
		{
			Constant content(std::move(location), std::move(type), std::forward<Args>(args)...);
			return std::unique_ptr<Expression>(new Expression(std::move(content)));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Expression> reference(SourceRange location, Type type, Args&&... args)
		{
			ReferenceAccess content(std::move(location), std::move(type), std::forward<Args>(args)...);
			return std::unique_ptr<Expression>(new Expression(std::move(content)));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Expression> operation(SourceRange location, Type type, OperationKind kind, Args&&... args)
		{
			Operation content(std::move(location), std::move(type), kind, std::forward<Args>(args)...);
			return std::unique_ptr<Expression>(new Expression(std::move(content)));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Expression> tuple(SourceRange location, Type type, Args&&... args)
		{
			Tuple content(std::move(location), std::move(type), std::forward<Args>(args)...);
			return std::unique_ptr<Expression>(new Expression(std::move(content)));
		}

		private:
		explicit Expression(Array content);
		explicit Expression(Call content);
		explicit Expression(Constant content);
		explicit Expression(Operation content);
		explicit Expression(ReferenceAccess content);
		explicit Expression(Tuple content);

		std::variant<Array, Call, Constant, Operation, ReferenceAccess, Tuple> content;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Expression& obj);

	std::string toString(const Expression& obj);
}
