#pragma once

#include <cassert>
#include <string>
#include <type_traits>

#include "Expression.h"
#include "Type.h"

namespace modelica::frontend
{
	class Constant
			: public impl::ExpressionCRTP<Constant>,
				public impl::Cloneable<Constant>
	{
		public:
		Constant(SourcePosition location, Type type, bool val);
		Constant(SourcePosition location, Type type, int val);
		Constant(SourcePosition location, Type type, float val);
		Constant(SourcePosition location, Type type, double val);
		Constant(SourcePosition location, Type type, char val);
		Constant(SourcePosition location, Type type, std::string val);

		Constant(const Constant& other);
		Constant(Constant&& other);
		~Constant() override;

		Constant& operator=(const Constant& other);
		Constant& operator=(Constant&& other);

		friend void swap(Constant& first, Constant& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::EXPRESSION_CONSTANT;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool isLValue() const override;

		[[nodiscard]] bool operator==(const Constant& other) const;
		[[nodiscard]] bool operator!=(const Constant& other) const;

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
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<frontendTypeToType_v<T>>(value);
		}

		template<BuiltInType T>
		[[nodiscard]] frontendTypeToType_v<T>& get()
		{
			assert(isA<T>());
			return std::get<frontendTypeToType_v<T>>(value);
		}

		template<BuiltInType T>
		[[nodiscard]] const frontendTypeToType_v<T>& get() const
		{
			assert(isA<T>());
			return std::get<frontendTypeToType_v<T>>(value);
		}

		template<BuiltInType T>
		[[nodiscard]] frontendTypeToType_v<T> as() const
		{
			using Tr = frontendTypeToType_v<T>;

			if (isA<BuiltInType::Integer>())
				return static_cast<Tr>(get<BuiltInType::Integer>());

			if (isA<BuiltInType::Float>())
				return static_cast<Tr>(get<BuiltInType::Float>());

			if (isA<BuiltInType::Boolean>())
				return static_cast<Tr>(get<BuiltInType::Boolean>());

			assert(false && "unreachable");
			return {};
		}

		private:
		std::variant<bool, int, double, std::string> value;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Constant& obj);

	std::string toString(const Constant& obj);
}
