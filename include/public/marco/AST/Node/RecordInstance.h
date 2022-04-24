#ifndef MARCO_AST_NODE_RECORDINSTANCE_H
#define MARCO_AST_NODE_RECORDINSTANCE_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::ast
{
  class Expression;

  class RecordInstance
      : public ASTNode,
        public impl::Dumpable<RecordInstance>
  {
    private:
      template<typename T> using Container = llvm::SmallVector<T, 3>;

    public:
      using iterator = Container<std::unique_ptr<Expression>>::iterator;
      using const_iterator = Container<std::unique_ptr<Expression>>::const_iterator;

      RecordInstance(const RecordInstance& other);
      RecordInstance(RecordInstance&& other);
      ~RecordInstance() override;

      RecordInstance& operator=(const RecordInstance& other);
      RecordInstance& operator=(RecordInstance&& other);

      [[nodiscard]] bool operator==(const RecordInstance& other) const;
      [[nodiscard]] bool operator!=(const RecordInstance& other) const;

      [[nodiscard]] bool isLValue() const;

      [[nodiscard]] Type& getType();
      [[nodiscard]] const Type& getType() const;
      void setType(Type tp);

      [[nodiscard]] const Record* getRecordType() const;

      friend void swap(RecordInstance& first, RecordInstance& second);

      void print(llvm::raw_ostream& os, size_t indents = 0) const override;

      [[nodiscard]] Expression& getMemberValue(llvm::StringRef name);

      [[nodiscard]] iterator begin();
      [[nodiscard]] const_iterator begin()const;

      [[nodiscard]] iterator end();
      [[nodiscard]] const_iterator end() const;

    private:
      friend class Expression;

      RecordInstance(SourceRange location,
                     Type type,
                     llvm::ArrayRef<std::unique_ptr<Expression>> values);

    private:
      const Record *recordType;
      Type type;
      Container<std::unique_ptr<Expression>> values;
  };

  llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const RecordInstance& obj);

  std::string toString(const RecordInstance& obj);
}

#endif // MARCO_AST_NODE_RECORDINSTANCE_H
