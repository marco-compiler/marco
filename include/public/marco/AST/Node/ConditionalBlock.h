#ifndef MARCO_AST_NODE_CONDITIONALBLOCK_H
#define MARCO_AST_NODE_CONDITIONALBLOCK_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/AST/Node/Expression.h"

namespace marco::ast
{
	template<typename T>
	class ConditionalBlock : public impl::Dumpable<ConditionalBlock<T>>
	{
		private:
      using Container = llvm::SmallVector<std::unique_ptr<T>, 3>;

		public:
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      ConditionalBlock(std::unique_ptr<Expression> condition,
                       llvm::ArrayRef<std::unique_ptr<T>> body)
          : condition(std::move(condition))
      {
        for (const auto& element : body)
          this->body.push_back(element->clone());
      }

      ConditionalBlock(const ConditionalBlock<T>& other)
          : condition(other.condition->clone())
      {
        for (const auto& element : other.body)
          this->body.push_back(element->clone());
      }

      ConditionalBlock(ConditionalBlock<T>&& other) = default;
      ~ConditionalBlock() = default;

      ConditionalBlock<T>& operator=(const ConditionalBlock<T>& other)
      {
        ConditionalBlock<T> result(other);
        swap(*this, result);
        return *this;
      }

      ConditionalBlock& operator=(ConditionalBlock&& other) = default;

      friend void swap(ConditionalBlock<T>& first, ConditionalBlock<T>& second)
      {
        using std::swap;
      }

      void print(llvm::raw_ostream& os, size_t indents = 0) const override
      {
        os.indent(indents);
        os << "condition:\n";
        condition->print(os, indents + 1);

        os.indent(indents);
        os << "body:\n";

        for (const auto& statement : body)
          statement->print(os, indents + 1);
      }

      [[nodiscard]] Expression* getCondition() { return condition.get(); }

      [[nodiscard]] const Expression* getCondition() const { return condition.get(); }

      [[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<T>> getBody() { return body; }

      [[nodiscard]] llvm::ArrayRef<std::unique_ptr<T>> getBody() const { return body; }

      void setBody(llvm::ArrayRef<std::unique_ptr<T>> elements)
      {
        body.clear();

        for (const auto& element : elements)
          body.push_back(element->clone());
      }

      [[nodiscard]] size_t size() const { return body.size(); }

      [[nodiscard]] iterator begin()
      {
        return body.begin();
      }

      [[nodiscard]] const_iterator begin() const
      {
        return body.begin();
      }

      [[nodiscard]] iterator end()
      {
        return body.end();
      }

      [[nodiscard]] const_iterator end() const
      {
        return body.end();
      }

		private:
      std::unique_ptr<Expression> condition;
      Container body;
	};
}

#endif // MARCO_AST_NODE_CONDITIONALBLOCK_H
