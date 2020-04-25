#pragma once

#include <initializer_list>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <modelica/frontend/Constant.hpp>
#include <modelica/frontend/Type.hpp>
#include <utility>
#include <variant>
#include <vector>

namespace modelica
{
	enum class OperationKind
	{
		add,
		subtract,
		multiply,
		divide,
		ifelse,
		subscription
	};

	[[nodiscard]] constexpr size_t arity(OperationKind kind)
	{
		switch (kind)
		{
			case OperationKind::add:
			case OperationKind::subtract:
			case OperationKind::multiply:
			case OperationKind::divide:
			case OperationKind::subscription:
				return 2;
			case OperationKind::ifelse:
				return 3;
		}
		return std::numeric_limits<size_t>::max();
	}

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
				assert(getArity() == arguments.size());
			}

			[[nodiscard]] OperationKind getKind() const { return kind; }
			void setKind(OperationKind k) { kind = k; }
			[[nodiscard]] size_t getArity() const { return arity(kind); }

			[[nodiscard]] bool operator==(const Operation& other) const;
			[[nodiscard]] bool operator!=(const Operation& other) const
			{
				return !(*this == other);
			}

			private:
			Container arguments;
			OperationKind kind;
		};

		template<OperationKind op, typename... Args>
		[[nodiscard]] Expression::Operation makeOp(Args&&... args)
		{
			static_assert(
					arity(op) == sizeof...(Args),
					"missmatch betweet arguments and operation arity");
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

		template<OperationKind op, typename... Args>
		explicit Expression(Type tp, Args&&... args)
				: content(makeOp<op>(std::forward<Args>(args)...)), type(std::move(tp))
		{
		}

		template<typename... Args>
		explicit Expression(Type tp, OperationKind kind, Args... args)
				: content(Operation(kind, std::forward<Args>(args)...)),
					type(std::move(tp))
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

		private:
		std::variant<Operation, Constant> content;
		Type type;
	};

}	 // namespace modelica
