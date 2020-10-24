#pragma once

#include <llvm/Support/raw_ostream.h>
#include <string>

namespace modelica
{
	/**
	 * A reference access is pretty much any use of a variable at the moment.
	 */
	class ReferenceAccess
	{
		public:
		ReferenceAccess(std::string name, bool globalLookup = false);

		[[nodiscard]] bool operator==(const ReferenceAccess& other) const;
		[[nodiscard]] bool operator!=(const ReferenceAccess& other) const;

		void dump() const;
		void dump(llvm::raw_ostream& os = llvm::outs(), size_t indents = 0) const;

		[[nodiscard]] std::string& getName();
		[[nodiscard]] const std::string& getName() const;
		[[nodiscard]] bool hasGlobalLookup() const;

		private:
		std::string referencedName;
		bool globalLookup;
	};
}	 // namespace modelica
