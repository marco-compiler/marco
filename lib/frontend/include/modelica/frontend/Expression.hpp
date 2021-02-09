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
		Expression(Type type, Constant constant);
		Expression(Type type, ReferenceAccess access);
		Expression(Type type, Operation operation);
		Expression(Type type, Call call);
		Expression(Type type, Tuple tuple);

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

		template<typename Arg>
		[[nodiscard]] static Expression constant(SourcePosition location, Type type, Arg&& arg)
		{
			Constant content(std::move(location), std::forward<Arg>(arg));
			return Expression(type, std::move(content));
		}

		template<typename... Args>
		[[nodiscard]] static Expression reference(SourcePosition location, Type type, Args&&... args)
		{
			ReferenceAccess content(location, std::forward<Args>(args)...);
			return Expression(type, std::move(content));
		}

		template<typename... Args>
		[[nodiscard]] static Expression operation(SourcePosition location, Type type, OperationKind kind, Args&&... args)
		{
			Operation content(location, kind, std::forward<Args>(args)...);
			return Expression(type, std::move(content));
		}

		template<typename... Args>
		[[nodiscard]] static Expression call(SourcePosition location, Type type, Expression function, Args&&... args)
		{
			Call content(location, std::move(function), std::forward<Args>(args)...);
			return Expression(type, std::move(content));
		}

		template<typename... Args>
		[[nodiscard]] static Expression tuple(SourcePosition location, Type type, Args&&... args)
		{
			Tuple content(location, std::forward<Args>(args)...);
			return Expression(type, std::move(content));
		}

		private:
		std::variant<Constant, ReferenceAccess, Operation, Call, Tuple> content;
		Type type;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Expression& obj);

	std::string toString(const Expression& obj);
}	 // namespace modelica
