#ifndef MARCO_AST_NODE_FOREQUATION_H
#define MARCO_AST_NODE_FOREQUATION_H

#include "marco/AST/Node/ASTNode.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast
{
	class Equation;
	class Induction;

	/// "For equations" are different with respect to regular equations
	/// because they introduce a set of inductions, and thus a new set of names
	/// available within the for cycle.
	///
	/// Inductions are mapped to a set of indexes so that an from a name we can
	/// deduce a index and from a index we can deduce a name.
	class ForEquation : public ASTNode
	{
		public:
      explicit ForEquation(SourceRange location);

      ForEquation(const ForEquation& other);

      ~ForEquation() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::ForEquation;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      size_t getNumOfInductions() const;

      Induction* getInduction(size_t index);

      const Induction* getInduction(size_t index) const;

      void setInductions(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

      void addOuterInduction(std::unique_ptr<ASTNode> node);

      Equation* getEquation() const;

      void setEquation(std::unique_ptr<ASTNode> node);

    private:
      llvm::SmallVector<std::unique_ptr<ASTNode>, 3> inductions;
      std::unique_ptr<ASTNode> equation;
	};
}

#endif // MARCO_AST_NODE_FOREQUATION_H
