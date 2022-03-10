#ifndef MARCO_AST_NODE_ARRAY_H
#define MARCO_AST_NODE_ARRAY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Type.h"
#include <memory>

namespace marco::ast
{
	class Expression;

	class Array
			: public ASTNode,
				public impl::Dumpable<Array>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::unique_ptr<Expression>>::iterator;
		using const_iterator = Container<std::unique_ptr<Expression>>::const_iterator;

		Array(const Array& other);
		Array(Array&& other);
		~Array() override;

		Array& operator=(const Array& other);
		Array& operator=(Array&& other);

		friend void swap(Array& first, Array& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool isLValue() const;

		[[nodiscard]] bool operator==(const Array& other) const;
		[[nodiscard]] bool operator!=(const Array& other) const;

		[[nodiscard]] Expression* operator[](size_t index);
		[[nodiscard]] const Expression* operator[](size_t index) const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type tp);

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		friend class Expression;

		Array(SourceRange location,
					Type type,
					llvm::ArrayRef<std::unique_ptr<Expression>> values);

		Type type;
		Container<std::unique_ptr<Expression>> values;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Array& obj);

	std::string toString(const Array& obj);
}

#endif // MARCO_AST_NODE_ARRAY_H
