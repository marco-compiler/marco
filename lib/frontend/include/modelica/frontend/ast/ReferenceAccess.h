#pragma once

#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourcePosition.h>
#include <string>

namespace modelica::frontend
{
	/**
	 * A reference access is pretty much any use of a variable at the moment.
	 */
	class ReferenceAccess
	{
		public:
		ReferenceAccess(SourcePosition location, std::string name, bool globalLookup = false, bool dummy = false);

		[[nodiscard]] bool operator==(const ReferenceAccess& other) const;
		[[nodiscard]] bool operator!=(const ReferenceAccess& other) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

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

		static ReferenceAccess dummy(SourcePosition location);

		private:
		SourcePosition location;
		std::string referencedName;
		bool globalLookup;
		bool dummyVariable;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const ReferenceAccess& obj);

	std::string toString(const ReferenceAccess& obj);
}
