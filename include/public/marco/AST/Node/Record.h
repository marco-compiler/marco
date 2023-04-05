#ifndef MARCO_AST_NODE_RECORD_H
#define MARCO_AST_NODE_RECORD_H

#include "marco/AST/Node/Class.h"

namespace marco::ast
{
  class VariableType;

  class RecordType
  {
    public:
      RecordType(llvm::ArrayRef<std::unique_ptr<ASTNode>> types);

      size_t size() const;

      const VariableType* getType(size_t index) const;

    private:
      llvm::SmallVector<std::unique_ptr<ASTNode>, 3> body;
  };

  class Record : public Class
  {
    public:
      explicit Record(SourceRange location);

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Class_Record;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      RecordType getType() const;
  };
}

#endif // MARCO_AST_NODE_RECORD_H
