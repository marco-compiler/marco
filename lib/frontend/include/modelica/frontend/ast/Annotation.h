#pragma once

#include "ASTNode.h"

namespace modelica::frontend
{
	class ClassModification;
	class InverseFunctionAnnotation;

	class Annotation
			: public ASTNode,
				public impl::Cloneable<Annotation>,
				public impl::Dumpable<Annotation>
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

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool getInlineProperty() const;
		[[nodiscard]] InverseFunctionAnnotation getInverseFunctionAnnotation() const;

		private:
		std::unique_ptr<ClassModification> properties;
	};
}
