#pragma once

#include <initializer_list>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "Call.hpp"
#include "Constant.hpp"
#include "ReferenceAccess.hpp"
#include "Type.hpp"

namespace modelica
{
	enum class OperationKind
	{
		empty,
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
			explicit Operation(OperationKind kind, Args&&... args)
					: arguments({ std::forward<Args>(args)... }), kind(kind)
			{
			}

			Operation(OperationKind kind, Container args);

			[[nodiscard]] bool operator==(const Operation& other) const;
			[[nodiscard]] bool operator!=(const Operation& other) const;

			[[nodiscard]] Expression& operator[](size_t index);
			[[nodiscard]] const Expression& operator[](size_t index) const;

			void dump() const;
			void dump(
					llvm::raw_ostream& OS = llvm::outs(), size_t nestLevel = 0) const;

			[[nodiscard]] OperationKind getKind() const;
			void setKind(OperationKind k);

			[[nodiscard]] Container& getArguments();
			[[nodiscard]] const Container& getArguments() const;
			[[nodiscard]] size_t argumentsCount() const;

			[[nodiscard]] Container::iterator begin();
			[[nodiscard]] Container::const_iterator begin() const;

			[[nodiscard]] Container::iterator end();
			[[nodiscard]] Container::const_iterator end() const;

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
		Expression(Type type, T&& constant)
				: content(Constant(std::forward<T>(constant))), type(std::move(type))
		{
		}

		Expression(Type tp, Constant costnt);
		Expression(Type tp, ReferenceAccess access);
		Expression(Type tp, Call call);

		template<OperationKind op, typename... Args>
		Expression(Type type, Args&&... args)
				: content(makeOp<op>(std::forward<Args>(args)...)),
					type(std::move(type))
		{
		}

		template<typename... Args>
		Expression(Type type, OperationKind kind, Args... args)
				: content(Operation(kind, std::forward<Args>(args)...)),
					type(std::move(type))
		{
		}

		Expression(Type tp, OperationKind kind, Operation::Container args);

		~Expression() = default;

		Expression(const Expression& other) = default;
		Expression(Expression&& other) = default;
		Expression& operator=(const Expression& other) = default;
		Expression& operator=(Expression&& other) = default;

		[[nodiscard]] bool operator==(const Expression& other) const;
		[[nodiscard]] bool operator!=(const Expression& other) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

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

		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<T>(content);
		}

		[[nodiscard]] bool isOperation() const;
		[[nodiscard]] Operation& getOperation();
		[[nodiscard]] const Operation& getOperation() const;
		[[nodiscard]] OperationKind getOperationKind() const;

		[[nodiscard]] Constant& getConstant();
		[[nodiscard]] const Constant& getConstant() const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type tp);

		[[nodiscard]] static Expression trueExp()
		{
			return Expression(makeType<bool>(), true);
		}

		[[nodiscard]] static Expression falseExp()
		{
			return Expression(makeType<bool>(), false);
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

		[[nodiscard]] static Expression add(Type t, Operation::Container args)
		{
			return op<OperationKind::add>(std::move(t), std::move(args));
		}

		template<typename... Args>
		[[nodiscard]] static Expression add(Type t, Args&&... args)
		{
			return op<OperationKind::add>(std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression subscription(Type t, Args&&... args)
		{
			return op<OperationKind::subscription>(
					std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression negate(Type t, Args&&... args)
		{
			return op<OperationKind::negate>(
					std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression subtract(Type t, Args&&... args)
		{
			return op<OperationKind::subtract>(
					std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression divide(Type t, Args&&... args)
		{
			return op<OperationKind::divide>(
					std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression powerOf(Type t, Args&&... args)
		{
			return op<OperationKind::powerOf>(
					std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression memberLookup(Type t, Args&&... args)
		{
			return op<OperationKind::memberLookup>(
					std::move(t), std::forward<Args>(args)...);
		}

		[[nodiscard]] static Expression multiply(Type t, Operation::Container args)
		{
			return op<OperationKind::multiply>(std::move(t), std::move(args));
		}

		[[nodiscard]] static Expression lor(Type t, Operation::Container args)
		{
			return op<OperationKind::lor>(std::move(t), std::move(args));
		}

		[[nodiscard]] static Expression land(Type t, Operation::Container args)
		{
			return op<OperationKind::land>(std::move(t), std::move(args));
		}

		private:
		std::variant<Operation, Constant, ReferenceAccess, Call> content;
		Type type;
	};

	[[nodiscard]] Expression makeCall(
			Expression fun, llvm::ArrayRef<Expression> args);
}	 // namespace modelica
