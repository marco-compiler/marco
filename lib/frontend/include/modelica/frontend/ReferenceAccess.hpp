#pragma once

#include <string>
namespace modelica
{
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

		private:
		std::string referencedName;
		bool globalLookup;
	};
}	 // namespace modelica
