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
		ReferenceAccess(
				std::string name, bool globalLookup = false, bool dummy = false);

		[[nodiscard]] bool operator==(const ReferenceAccess& other) const;
		[[nodiscard]] bool operator!=(const ReferenceAccess& other) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] std::string& getName();
		[[nodiscard]] const std::string& getName() const;
		void setName(std::string name);

		[[nodiscard]] bool hasGlobalLookup() const;

		/**
		 * Get whether the referenced variable is created just for temporary
		 * use (such as a function output that is then discarded) and thus the
		 * reference points to a not already existing variable.
		 *
		 * @return true if temporary; false otherwise
		 */
		[[nodiscard]] bool isDummy() const;

		static ReferenceAccess dummy();

		private:
		std::string referencedName;
		bool globalLookup;
		bool dummyVariable;
	};
}	 // namespace modelica
