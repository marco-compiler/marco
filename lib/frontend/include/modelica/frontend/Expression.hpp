#pragma once

#include <initializer_list>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <modelica/frontend/Call.hpp>
#include <modelica/frontend/Constant.hpp>
#include <modelica/frontend/Operation.hpp>
#include <modelica/frontend/ReferenceAccess.hpp>
#include <modelica/frontend/Tuple.hpp>
#include <modelica/frontend/Type.hpp>
#include <utility>
#include <variant>
#include <vector>

namespace modelica
{
	class Expression
	{
		public:
		template<typename T>
		Expression(SourcePosition location, Type type, T&& constant) : location(std::move(location)), content(Constant(std::forward<T>(constant))), type(std::move(type))
		{
		}

		Expression(SourcePosition location, Type tp, Constant costnt);
		Expression(SourcePosition location, Type tp, ReferenceAccess access);
		Expression(SourcePosition location, Type tp, Call call);
		Expression(SourcePosition location, Type tp, Tuple tuple);

		template<OperationKind op, typename... Args>
		Expression(SourcePosition location, Type type, Args&&... args) : location(std::move(location)), content(makeOp<op>(std::forward<Args>(args)...)), type(std::move(type))
		{
		}

		template<typename... Args>
		Expression(SourcePosition location, Type type, OperationKind kind, Args... args) : location(std::move(location)), content(Operation(kind, std::forward<Args>(args)...)), type(std::move(type))
		{
		}

		Expression(SourcePosition location, Type tp, OperationKind kind, Operation::Container args);

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

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] bool isLValue() const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type tp);

		[[nodiscard]] static Expression trueExp(SourcePosition location)
		{
			return Expression(location, makeType<bool>(), true);
		}

		[[nodiscard]] static Expression falseExp(SourcePosition location)
		{
			return Expression(location, makeType<bool>(), false);
		}

		template<OperationKind o, typename... Args>
		[[nodiscard]] static Expression op(SourcePosition location, Type tp, Args&&... args)
		{
			return Expression(location, std::move(tp), o, std::forward<Args>(args)...);
		}

		template<OperationKind o>
		[[nodiscard]] static Expression op(SourcePosition location, Type tp, Operation::Container args)
		{
			return Expression(location, std::move(tp), o, std::move(args));
		}

		[[nodiscard]] static Expression add(SourcePosition location, Type t, Operation::Container args)
		{
			return op<OperationKind::add>(location, std::move(t), std::move(args));
		}

		template<typename... Args>
		[[nodiscard]] static Expression add(SourcePosition location, Type t, Args&&... args)
		{
			return op<OperationKind::add>(location, std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression subscription(SourcePosition location, Type t, Args&&... args)
		{
			return op<OperationKind::subscription>(
					location, std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression negate(SourcePosition location, Type t, Args&&... args)
		{
			return op<OperationKind::negate>(
					location, std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression subtract(SourcePosition location, Type t, Args&&... args)
		{
			return op<OperationKind::subtract>(
					location, std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression divide(SourcePosition location, Type t, Args&&... args)
		{
			return op<OperationKind::divide>(
					location, std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression powerOf(SourcePosition location, Type t, Args&&... args)
		{
			return op<OperationKind::powerOf>(
					location, std::move(t), std::forward<Args>(args)...);
		}

		template<typename... Args>
		[[nodiscard]] static Expression memberLookup(SourcePosition location, Type t, Args&&... args)
		{
			return op<OperationKind::memberLookup>(
					location, std::move(t), std::forward<Args>(args)...);
		}

		[[nodiscard]] static Expression multiply(SourcePosition location, Type t, Operation::Container args)
		{
			return op<OperationKind::multiply>(location, std::move(t), std::move(args));
		}

		[[nodiscard]] static Expression lor(SourcePosition location, Type t, Operation::Container args)
		{
			return op<OperationKind::lor>(location, std::move(t), std::move(args));
		}

		[[nodiscard]] static Expression land(SourcePosition location, Type t, Operation::Container args)
		{
			return op<OperationKind::land>(location, std::move(t), std::move(args));
		}

		private:
		SourcePosition location;
		std::variant<Operation, Constant, ReferenceAccess, Call, Tuple> content;
		Type type;
	};

	[[nodiscard]] Expression makeCall(
			SourcePosition location, Expression fun, llvm::ArrayRef<Expression> args);

	template<typename... Expressions>
	[[nodiscard]] Expression makeTuple(SourcePosition location, Expressions&&... expressions)
	{
		return Expression(Tuple(std::forward<Expressions>(expressions)...));
	}
}	 // namespace modelica
