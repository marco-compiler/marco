#ifndef MARCO_AST_NODE_ANNOTATION_H
#define MARCO_AST_NODE_ANNOTATION_H

#include "marco/AST/Node/ASTNode.h"

namespace marco::ast
{
	class ClassModification;
	class DerivativeAnnotation;
	class InverseFunctionAnnotation;

	class Annotation
			: public ASTNode,
				public impl::Cloneable<Annotation>,
				public impl::Dumpable<Annotation>
	{
		public:
		explicit Annotation(SourceRange location);
		Annotation(SourceRange location, std::unique_ptr<ClassModification> properties);

		Annotation(const Annotation& other);
		Annotation(Annotation&& other);
		~Annotation() override;

		Annotation& operator=(const Annotation& other);
		Annotation& operator=(Annotation&& other);

		friend void swap(Annotation& first, Annotation& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool getInlineProperty() const;

		[[nodiscard]] bool hasDerivativeAnnotation() const;
		[[nodiscard]] DerivativeAnnotation getDerivativeAnnotation() const;

		[[nodiscard]] InverseFunctionAnnotation getInverseFunctionAnnotation() const;

		private:
		std::unique_ptr<ClassModification> properties;
	};
}

#endif // MARCO_AST_NODE_ANNOTATION_H
