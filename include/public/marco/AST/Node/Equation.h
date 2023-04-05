#ifndef MARCO_AST_NODE_EQUATION_H
#define MARCO_AST_NODE_EQUATION_H

#include "marco/AST/Node/ASTNode.h"
#include <memory>

namespace marco::ast
{
	class Expression;

	class Equation : public ASTNode
	{
		public:
      explicit Equation(SourceRange location);

      Equation(const Equation& other);

      ~Equation() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Equation;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      Expression* getLhsExpression();

      const Expression* getLhsExpression() const;

      void setLhsExpression(std::unique_ptr<ASTNode> node);

      Expression* getRhsExpression();

      const Expression* getRhsExpression() const;

      void setRhsExpression(std::unique_ptr<ASTNode> node);

    private:
      std::unique_ptr<ASTNode> lhs;
      std::unique_ptr<ASTNode> rhs;
	};
}

#endif // MARCO_AST_NODE_EQUATION_H
