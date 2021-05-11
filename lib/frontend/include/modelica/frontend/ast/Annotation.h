#pragma once

#include <llvm/ADT/Optional.h>

#include "ASTNode.h"

namespace modelica::frontend
{
	class ClassModification;
	class InverseFunctionAnnotation;

	class Annotation
			: public impl::ASTNodeCRTP<Annotation>,
				public impl::Cloneable<Annotation>
	{
		public:
		explicit Annotation(SourcePosition location);
		Annotation(SourcePosition location, std::unique_ptr<ClassModification> properties);

		Annotation(const Annotation& other);
		Annotation(Annotation&& other);
		~Annotation() override;

		Annotation& operator=(const Annotation& other);
		Annotation& operator=(Annotation&& other);

		friend void swap(Annotation& first, Annotation& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::ANNOTATION;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool getInlineProperty() const;
		[[nodiscard]] InverseFunctionAnnotation getInverseFunctionAnnotation() const;

		private:
		std::unique_ptr<ClassModification> properties;
	};
}
