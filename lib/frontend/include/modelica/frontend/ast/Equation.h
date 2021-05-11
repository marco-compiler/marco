#pragma once

#include <memory>

#include "ASTNode.h"

namespace modelica::frontend
{
	class Expression;

	class Equation
			: public impl::ASTNodeCRTP<Equation>,
				public impl::Cloneable<Equation>
	{
		public:
		Equation(SourcePosition location,
						 std::unique_ptr<Expression> lhs,
						 std::unique_ptr<Expression> rhs);

		Equation(const Equation& other);
		Equation(Equation&& other);
		~Equation() override;

		Equation& operator=(const Equation& other);
		Equation& operator=(Equation&& other);

		friend void swap(Equation& first, Equation& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::EQUATION;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] Expression* getLhsExpression();
		[[nodiscard]] const Expression* getLhsExpression() const;
		void setLhsExpression(Expression* expression);

		[[nodiscard]] Expression* getRhsExpression();
		[[nodiscard]] const Expression* getRhsExpression() const;
		void setRhsExpression(Expression* expression);

		private:
		std::unique_ptr<Expression> lhs;
		std::unique_ptr<Expression> rhs;
	};
}
