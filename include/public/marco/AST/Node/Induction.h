#ifndef MARCO_AST_NODE_INDUCTION_H
#define MARCO_AST_NODE_INDUCTION_H

#include "marco/AST/Node/ASTNode.h"
#include <memory>

namespace marco::ast
{
	class Expression;

	/// An induction is the in memory of a piece of code such as
	/// for i in 0:20. and induction holds a name and the begin and end
	/// expressions.
	///
	/// Notice that for the compiler we made the assumption that all range will be
	/// of step one.
	class Induction : public ASTNode
	{
		public:
      explicit Induction(SourceRange location);

      Induction(const Induction& other);

      ~Induction() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Induction;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      llvm::StringRef getName() const;

      void setName(llvm::StringRef newName);

      Expression* getBegin();

      const Expression* getBegin() const;

      void setBegin(std::unique_ptr<ASTNode> node);

      Expression* getEnd();

      const Expression* getEnd() const;

      void setEnd(std::unique_ptr<ASTNode> node);

      Expression* getStep();

      const Expression* getStep() const;

      void setStep(std::unique_ptr<ASTNode> node);

      size_t getInductionIndex() const;

      void setInductionIndex(size_t index);

    private:
      std::string inductionVariable;
      std::unique_ptr<ASTNode> begin;
      std::unique_ptr<ASTNode> end;
      std::unique_ptr<ASTNode> step;
      size_t inductionIndex;
	};
}

#endif // MARCO_AST_NODE_INDUCTION_H
