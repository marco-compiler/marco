#pragma once

#include <mlir/IR/OpDefinition.h>

namespace modelica
{


	class FunctionOp
	{
		public:
		//using Op::Op;

		static llvm::StringRef getOperationName();
	};


}