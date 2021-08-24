#pragma once

#include <llvm/Support/Error.h>

namespace marco::frontend
{
	class Class;

	class Pass
	{
		public:
		Pass() = default;
		Pass(const Pass& other) = default;

		Pass(Pass&& other) = default;
		Pass& operator=(Pass&& other) = default;

		virtual ~Pass() = default;

		Pass& operator=(const Pass& other) = default;

		virtual llvm::Error run(llvm::ArrayRef<std::unique_ptr<Class>> classes) = 0;
	};
}
