#pragma once

#include <llvm/Support/Error.h>

namespace modelica
{
	class ClassContainer;

	class Pass
	{
		public:
		virtual llvm::Error run(ClassContainer& cls) = 0;
	};
}
