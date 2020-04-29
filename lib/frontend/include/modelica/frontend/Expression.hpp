#pragma once

#include <initializer_list>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <modelica/frontend/Call.hpp>
#include <modelica/frontend/Constant.hpp>
#include <modelica/frontend/Type.hpp>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "modelica/frontend/ReferenceAccess.hpp"

namespace modelica
{
	enum class OperationKind
	{
		negate,
		add,
		subtract,
		multiply,
		divide,
		ifelse,
		greater,
		greaterEqual,
		equal,
		different,
		lessEqual,
		less,
		land,
		lor,
		subscription,
		memberLookup,
		powerOf,
	};

	class Expression
	{
		public:
		class Operation
		{
			public:
			using Container = std::vector<Expression>;

			template<typename... Args>
			explicit Operation(OperationKind kind, Args... args)
					: arguments({ std::forward<Args>(args)... }), kind(kind)
			{
			}

			Operation(OperationKind kind, Container args)
					: arguments(std::move(args)), kind(kind)
			{
			}

			[[nodiscard]] OperationKind getKind() const { return kind; }
			[[nodiscard]] const Expression& operator[](size_t index) const
			{
				return arguments[index];
			}

			[[nodiscard]] Expression& operator[](size_t index)
			{
				return arguments[index];
			}
			void setKind(OperationKind k) { kind = k; }

			[[nodiscard]] size_t argumentsCount() const { return arguments.size(); }

			[[nodiscard]] bool operator==(const Operation& other) const;
			[[nodiscard]] bool operator!=(const Operation& other) const
			{
				return !(*this == other);
			}

			void dump(
					llvm::raw_ostream& OS = llvm::outs(), size_t nestLevel = 0) const;

			[[nodiscard]] auto begin() const { return arguments.begin(); }
			[[nodiscard]] auto begin() { return arguments.begin(); }
			[[nodiscard]] auto end() const { return arguments.end(); }
			[[nodiscard]] auto end() { return arguments.end(); }
			[[nodiscard]] const Container& getArguments() const { return arguments; }
			[[nodiscard]] Container& getArguments() { return arguments; }

			private:
			Container arguments;
			OperationKind kind;
		};

		template<OperationKind op, typename... Args>
		[[nodiscard]] Expression::Operation makeOp(Args&&... args)
		{
			return Operation(op, std::forward(args)...);
		}

		template<typename T>
		explicit Expression(Type tp, T&& costnt)
				: content(Constant(std::forward<T>(costnt))), type(std::move(tp))
		{
		}

		Expression(Type tp, Constant costnt)
				: content(std::move(costnt)), type(std::move(tp))
		{
		}

		Expression(Type tp, ReferenceAccess access)
				: content(std::move(access)), type(std::move(tp))
		{
		}

		Expression(Type tp, Call call)
				: content(std::move(call)), type(std::move(tp))
		{
		}

		template<OperationKind op, typename... Args>
		explicit Expression(Type tp, Args&&... args)
				: content(makeOp<op>(std::forward<Args>(args)...)), type(std::move(tp))
		{
		}

		template<typename... Args>
		Expression(Type tp, OperationKind kind, Args... args)
				: content(Operation(kind, std::forward<Args>(args)...)),
					type(std::move(tp))
		{
		}

		Expression(Type tp, OperationKind kind, Operation::Container args)
				: content(Operation(kind, std::move(args))), type(std::move(tp))
		{
		}

		~Expression() = default;
		Expression(const Expression& other) = default;
		Expression(Expression&& other) = default;
		Expression& operator=(const Expression& other) = default;
		Expression& operator=(Expression&& other) = default;
		[[nodiscard]] bool operator==(const Expression& other) const
		{
			return type == other.type && content == other.content;
		}

		[[nodiscard]] bool operator!=(const Expression& other) const
		{
			return !(*this == other);
		}

		[[nodiscard]] bool isOperation() const { return isA<Operation>(); }

		[[nodiscard]] Operation& getOperation() { return get<Operation>(); }

		[[nodiscard]] const Operation& getOperation() const
		{
			return get<Operation>();
		}

		[[nodiscard]] static Expression trueExp()
		{
			return Expression(makeType<bool>(), true);
		}

		[[nodiscard]] static Expression falseExp()
		{
			return Expression(makeType<bool>(), true);
		}

		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<T>(content);
		}

		[[nodiscard]] Constant& getConstant() { return get<Constant>(); }
		[[nodiscard]] const Constant& getConstant() const
		{
			return get<Constant>();
		}

		[[nodiscard]] const Type& getType() const { return type; }
		[[nodiscard]] Type& getType() { return type; }

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

		template<OperationKind o, typename... Args>
		[[nodiscard]] static Expression op(Type tp, Args&&... args)
		{
			return Expression(std::move(tp), o, std::forward<Args>(args)...);
		}

		template<OperationKind o>
		[[nodiscard]] static Expression op(Type tp, Operation::Container args)
		{
			return Expression(std::move(tp), o, std::move(args));
		}

		void dump(llvm::raw_ostream& OS = llvm::outs(), size_t nestLevel = 0) const;
		void setType(Type tp) { type = std::move(tp); }

		private:
		std::variant<Operation, Constant, ReferenceAccess, Call> content;
		Type type;
	};

	template<typename... Args>
	[[nodiscard]] Expression makeCall(Expression fun, Args... arguments)
	{
		return Expression(
				Type::unkown(),
				Call(
						std::make_unique<Expression>(std::move(fun)),
						{ std::make_unique<Expression>(std::move(arguments))... }));
	}

	[[nodiscard]] Expression makeCall(
			Expression fun, llvm::SmallVector<Expression, 3> exps);

}	 // namespace modelica
