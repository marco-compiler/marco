#pragma once

#include <cassert>
#include <string>
#include <type_traits>

#include "ASTNode.h"
#include "Type.h"

namespace marco::frontend
{
	class Constant
			: public ASTNode,
				public impl::Dumpable<Constant>
	{
		public:
		Constant(const Constant& other);
		Constant(Constant&& other);
		~Constant() override;

		Constant& operator=(const Constant& other);
		Constant& operator=(Constant&& other);

		friend void swap(Constant& first, Constant& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool isLValue() const;

		[[nodiscard]] bool operator==(const Constant& other) const;
		[[nodiscard]] bool operator!=(const Constant& other) const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type tp);

		template<class Visitor>
		auto visit(Visitor&& vis)
		{
			return std::visit(std::forward<Visitor>(vis), value);
		}

		template<class Visitor>
		auto visit(Visitor&& vis) const
		{
			return std::visit(std::forward<Visitor>(vis), value);
		}

		template<BuiltInType T>
		[[nodiscard]] bool isa() const
		{
			return std::holds_alternative<frontendTypeToType_v<T>>(value);
		}

		template<BuiltInType T>
		[[nodiscard]] frontendTypeToType_v<T>& get()
		{
			assert(isa<T>());
			return std::get<frontendTypeToType_v<T>>(value);
		}

		template<BuiltInType T>
		[[nodiscard]] const frontendTypeToType_v<T>& get() const
		{
			assert(isa<T>());
			return std::get<frontendTypeToType_v<T>>(value);
		}

		template<BuiltInType T>
		[[nodiscard]] frontendTypeToType_v<T> as() const
		{
			using Tr = frontendTypeToType_v<T>;

			if (isa<BuiltInType::Integer>())
				return static_cast<Tr>(get<BuiltInType::Integer>());

			if (isa<BuiltInType::Float>())
				return static_cast<Tr>(get<BuiltInType::Float>());

			if (isa<BuiltInType::Boolean>())
				return static_cast<Tr>(get<BuiltInType::Boolean>());

			assert(false && "unreachable");
			return {};
		}

		private:
		friend class Expression;

		Constant(SourceRange location, Type type, bool value);
		Constant(SourceRange location, Type type, long value);
		Constant(SourceRange location, Type type, double value);
		Constant(SourceRange location, Type type, std::string value);

		Constant(SourceRange location, Type type, int value);
		Constant(SourceRange location, Type type, float value);

		Type type;
		std::variant<bool, long, double, std::string> value;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Constant& obj);

	std::string toString(const Constant& obj);
}
