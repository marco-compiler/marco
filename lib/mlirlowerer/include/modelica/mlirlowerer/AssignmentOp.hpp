#pragma once

#include <mlir/IR/OpDefinition.h>

namespace modelica
{


	class AssignmentOp
	{
		public:
		//using Op::Op;

		static llvm::StringRef getOperationName();
	};


}