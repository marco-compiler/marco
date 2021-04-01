#pragma once

#include <llvm/ADT/Optional.h>

namespace modelica::frontend
{
	class ClassModification;

	class Annotation
	{
		public:
		Annotation();
		explicit Annotation(ClassModification properties);

		bool getInlineProperty();

		private:
		std::shared_ptr<ClassModification> properties;
	};
}
