#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>

#include "Expression.h"

namespace modelica::frontend
{
	class Array
			: public impl::ExpressionCRTP<Array>,
				public impl::Cloneable<Array>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::unique_ptr<Expression>>::iterator;
		using const_iterator = Container<std::unique_ptr<Expression>>::const_iterator;

		Array(SourcePosition location,
					llvm::ArrayRef<std::unique_ptr<Expression>> values,
					Type type);

		Array(const Array& other);
		Array(Array&& other);
		~Array() override;

		Array& operator=(const Array& other);
		Array& operator=(Array&& other);

		friend void swap(Array& first, Array& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::EXPRESSION_ARRAY;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool isLValue() const override;

		[[nodiscard]] bool operator==(const Array& other) const;
		[[nodiscard]] bool operator!=(const Array& other) const;

		[[nodiscard]] Expression* operator[](size_t index);
		[[nodiscard]] const Expression* operator[](size_t index) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		Container<std::unique_ptr<Expression>> values;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Array& obj);

	std::string toString(const Array& obj);
}
