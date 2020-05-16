#pragma once

#include <string>

#include "llvm/Support/raw_ostream.h"
namespace modelica
{
	/**
	 * A reference access is pretty much any use of a variable at the moment
	 */
	class ReferenceAccess
	{
		public:
		explicit ReferenceAccess(std::string name, bool globalLookup = false)
				: referencedName(std::move(name)), globalLookup(globalLookup)
		{
		}

		[[nodiscard]] const std::string& getName() const { return referencedName; }
		[[nodiscard]] std::string& getName() { return referencedName; }
		[[nodiscard]] bool hasGlobalLookup() const { return globalLookup; }
		[[nodiscard]] bool operator==(const ReferenceAccess& other) const
		{
			return globalLookup == other.globalLookup &&
						 referencedName == other.referencedName;
		}

		[[nodiscard]] bool operator!=(const ReferenceAccess& other) const
		{
			return !(*this == other);
		}
		void dump(
				llvm::raw_ostream& OS = llvm::outs(), size_t indentLevel = 0) const
		{
			OS.indent(indentLevel);
			OS << "reference access " << (globalLookup ? "." : "") << referencedName;
		}

		private:
		std::string referencedName;
		bool globalLookup;
	};
}	 // namespace modelica
