#pragma once

#include <llvm/Support/Error.h>

namespace modelica::frontend
{
	class ClassContainer;

	class Pass
	{
		public:
		virtual llvm::Error run(ClassContainer& cls) = 0;
	};
}
