#pragma once

#include <llvm/ADT/Optional.h>

namespace modelica::frontend
{
	class ClassModification;
	class InverseFunctionAnnotation;

	class Annotation
	{
		public:
		Annotation();
		explicit Annotation(ClassModification properties);

		[[nodiscard]] bool getInlineProperty() const;
		[[nodiscard]] InverseFunctionAnnotation getInverseFunctionAnnotation() const;

		private:
		std::shared_ptr<ClassModification> properties;
	};
}
