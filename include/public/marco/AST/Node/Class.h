#ifndef MARCO_AST_NODE_CLASS_H
#define MARCO_AST_NODE_CLASS_H

#include "marco/AST/Node/ASTNode.h"

namespace marco::ast
{
	class Class : public ASTNode
	{
		public:
      using ASTNode::ASTNode;

      Class(const Class& other);

      virtual ~Class();

      static bool classof(const ASTNode* node)
      {
        return node->getKind() >= ASTNode::Kind::Class &&
          node->getKind() <= ASTNode::Kind::Class_LastClass;
      }

    protected:
      virtual void addJSONProperties(llvm::json::Object& obj) const override;

    public:
      /// Get the name.
      llvm::StringRef getName() const;

      /// Set the name.
      void setName(llvm::StringRef newName);

      /// Get the variables.
      llvm::ArrayRef<std::unique_ptr<ASTNode>> getVariables() const;

      /// Set the variables.
      void setVariables(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

      /// Get the 'equations' blocks.
      llvm::ArrayRef<std::unique_ptr<ASTNode>> getEquationsBlocks() const;

      /// Set the 'equations' blocks.
      void setEquationsBlocks(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

      /// Get the 'initial equations' blocks.
      llvm::ArrayRef<std::unique_ptr<ASTNode>>
      getInitialEquationsBlocks() const;

      /// Set the 'initial equations' blocks.
      void setInitialEquationsBlocks(
          llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

      /// Get the 'algorithm' blocks.
      llvm::ArrayRef<std::unique_ptr<ASTNode>> getAlgorithms() const;

      /// Set the 'algorithm' blocks.
      void setAlgorithms(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

      /// Get the inner classes.
      llvm::ArrayRef<std::unique_ptr<ASTNode>> getInnerClasses() const;

      /// Set the inner classes.
      void setInnerClasses(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

    private:
      std::string name;
      llvm::SmallVector<std::unique_ptr<ASTNode>> variables;
      llvm::SmallVector<std::unique_ptr<ASTNode>> equationsBlocks;
      llvm::SmallVector<std::unique_ptr<ASTNode>> initialEquationsBlocks;
      llvm::SmallVector<std::unique_ptr<ASTNode>> algorithms;
      llvm::SmallVector<std::unique_ptr<ASTNode>> innerClasses;
	};
}

#endif // MARCO_AST_NODE_CLASS_H
