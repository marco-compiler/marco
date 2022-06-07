#ifndef MARCO_AST_NODE_ALGORITHM_H
#define MARCO_AST_NODE_ALGORITHM_H

#include "marco/AST/Node/ASTNode.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::ast
{
	class Statement;

	class Algorithm
			: public ASTNode,
				public impl::Cloneable<Algorithm>,
				public impl::Dumpable<Algorithm>
	{
		private:
		  template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
      using statements_iterator = Container<std::unique_ptr<Statement>>::iterator;
      using statements_const_iterator = Container<std::unique_ptr<Statement>>::const_iterator;

      template<typename... Args>
      static std::unique_ptr<Algorithm> build(Args&&... args)
      {
        return std::unique_ptr<Algorithm>(new Algorithm(std::forward<Args>(args)...));
      }

      Algorithm(const Algorithm& other);
      Algorithm(Algorithm&& other);

      ~Algorithm() override;

      Algorithm& operator=(const Algorithm& other);
      Algorithm& operator=(Algorithm&& other);

      friend void swap(Algorithm& first, Algorithm& second);

      void print(llvm::raw_ostream& os, size_t indents = 0) const override;

      Statement* operator[](size_t index);
      const Statement* operator[](size_t index) const;

      [[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Statement>> getBody();
      [[nodiscard]] llvm::ArrayRef<std::unique_ptr<Statement>> getBody() const;

      void setBody(llvm::ArrayRef<std::unique_ptr<Statement>> body);

      [[nodiscard]] size_t size() const;
      [[nodiscard]] bool empty() const;

      [[nodiscard]] statements_iterator begin();
      [[nodiscard]] statements_const_iterator begin() const;

      [[nodiscard]] statements_iterator end();
      [[nodiscard]] statements_const_iterator end() const;

		private:
      Algorithm(SourceRange location, llvm::ArrayRef<std::unique_ptr<Statement>> statements);

    private:
      Container<std::unique_ptr<Statement>> statements;
	};
}

#endif // MARCO_AST_NODE_ALGORITHM_H
