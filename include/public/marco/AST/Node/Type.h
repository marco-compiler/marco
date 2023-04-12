#ifndef MARCO_AST_NODE_TYPE_H
#define MARCO_AST_NODE_TYPE_H

#include "marco/AST/Node/ASTNode.h"
#include "llvm/ADT/STLExtras.h"
#include <string>

namespace marco::ast
{
  class ArrayDimension;

  class VariableType : public ASTNode
  {
    public:
      using ASTNode::ASTNode;

      VariableType(const VariableType& other);

      ~VariableType() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() >= ASTNode::Kind::VariableType &&
            node->getKind() <= ASTNode::Kind::VariableType_LastVariableType;
      }

      virtual void addJSONProperties(llvm::json::Object& obj) const override;

      size_t getRank() const;

      ArrayDimension* operator[](size_t index);

      const ArrayDimension* operator[](size_t index) const;

      llvm::ArrayRef<std::unique_ptr<ASTNode>> getDimensions() const;

      void setDimensions(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

      bool hasConstantShape() const;

      bool isScalar() const;

      std::unique_ptr<ASTNode> subscript(size_t times) const;

    private:
      llvm::SmallVector<std::unique_ptr<ASTNode>, 3> dimensions;
  };

  class BuiltInType : public VariableType
  {
    public:
      enum class Kind
      {
        Boolean,
        Integer,
        Real,
        String
      };

      BuiltInType(SourceRange location);

      BuiltInType(const BuiltInType& other);

      ~BuiltInType();

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::VariableType_BuiltIn;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      Kind getBuiltInTypeKind() const;

      void setBuiltInTypeKind(Kind newKind);

      bool isNumeric() const;

    private:
      Kind kind;
  };

  class UserDefinedType : public VariableType
  {
    public:
      UserDefinedType(SourceRange location);

      UserDefinedType(const UserDefinedType& other);

      ~UserDefinedType();

      static bool classof(const ASTNode* node)
      {
          return node->getKind() == ASTNode::Kind::VariableType_UserDefined;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      bool isGlobalLookup() const;

      void setGlobalLookup(bool global);

      size_t getPathLength() const;

      llvm::StringRef getElement(size_t index) const;

      void setPath(llvm::ArrayRef<std::string> newPath);

    private:
      bool globalLookup;
      llvm::SmallVector<std::string> path;
  };
}

#endif // MARCO_AST_NODE_TYPE_H
