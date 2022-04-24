#ifndef MARCO_AST_NODE_RECORD_H
#define MARCO_AST_NODE_RECORD_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Function.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::ast
{
	class Member;

	class Record
			: public ASTNode,
				public impl::Dumpable<Record>
	{
		private:
      template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
      using iterator = Container<std::unique_ptr<Member>>::iterator;
      using const_iterator = Container<std::unique_ptr<Member>>::const_iterator;

      Record(const Record& other);
      Record(Record&& other);
      ~Record() override;

      Record& operator=(const Record& other);
      Record& operator=(Record&& other);

      friend void swap(Record& first, Record& second);

      void print(llvm::raw_ostream& os, size_t indents = 0) const override;

      [[nodiscard]] bool operator==(const Record& other) const;
      [[nodiscard]] bool operator!=(const Record& other) const;

      [[nodiscard]] Member* operator[](llvm::StringRef name);
      [[nodiscard]] const Member* operator[](llvm::StringRef name) const;

      [[nodiscard]] llvm::StringRef getName() const;

      [[nodiscard]] size_t size() const;

      [[nodiscard]] iterator begin();
      [[nodiscard]] const_iterator begin()const;

      [[nodiscard]] iterator end();
      [[nodiscard]] const_iterator end() const;

      [[nodiscard]] bool shouldBeInlined() const;

      [[nodiscard]] const StandardFunction& getDefaultConstructor() const;

      void setAsNotInlineable();

		private:
      friend class Class;

      Record(SourceRange location,
             llvm::StringRef name,
             llvm::ArrayRef<std::unique_ptr<Member>> members);

      void setupDefaultConstructor();

    private:
      std::string name;
      Container<std::unique_ptr<Member>> members;

      std::unique_ptr<StandardFunction> defaultConstructor;
      bool inlineable=true;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Record& obj);

	std::string toString(const Record& obj);
}

#endif // MARCO_AST_NODE_RECORD_H
