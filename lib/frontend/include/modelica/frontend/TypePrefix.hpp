#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <type_traits>

namespace modelica
{
	enum class ParameterQualifier
	{
		discrete,
		parameter,
		constant,
		none
	};

	enum class IOQualifier
	{
		input,
		output,
		none
	};

	class TypePrefix
	{
		public:
		TypePrefix(ParameterQualifier parameterQualifier, IOQualifier ioQualifier);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		bool isParameter();

		private:
		ParameterQualifier parameterQualifier;
		IOQualifier ioQualifier;
	};
}	 // namespace modelica
