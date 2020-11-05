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

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const ParameterQualifier& obj);
	std::string toString(ParameterQualifier qualifier);

	enum class IOQualifier
	{
		input,
		output,
		none
	};

	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const IOQualifier& obj);
	std::string toString(IOQualifier qualifier);

	class TypePrefix
	{
		public:
		TypePrefix(ParameterQualifier parameterQualifier, IOQualifier ioQualifier);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] bool isParameter() const;
		[[nodiscard]] bool isInput() const;
		[[nodiscard]] bool isOutput() const;

		static TypePrefix empty();

		private:
		ParameterQualifier parameterQualifier;
		IOQualifier ioQualifier;
	};
}	 // namespace modelica
