#pragma once

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "modelica/frontend/Equation.hpp"
#include "modelica/frontend/Member.hpp"

namespace modelica
{
	class Class
	{
		public:
		Class(
				std::string name,
				llvm::SmallVector<Member, 3> memb,
				llvm::SmallVector<Equation, 3> equs)
				: name(std::move(name)),
					members(std::move(memb)),
					equations(std::move(equs))
		{
		}
		[[nodiscard]] const std::string& getName() const { return name; }
		[[nodiscard]] std::string& getName() { return name; }
		[[nodiscard]] const auto& getMembers() const { return members; }
		[[nodiscard]] auto& getMembers() { return members; }
		[[nodiscard]] size_t membersCount() const { return members.size(); }
		[[nodiscard]] const auto& getEquations() const { return equations; }

		[[nodiscard]] auto& getEquations() { return equations; }
		void addMember(Member newMember)
		{
			return members.push_back(std::move(newMember));
		}

		template<typename... Args>
		void emplaceMember(Args&&... args)
		{
			members.emplace_back(std::forward<Args>(args)...);
		}

		void eraseMember(size_t memberIndex)
		{
			assert(memberIndex < members.size());
			members.erase(members.begin() + memberIndex);
		}

		private:
		std::string name;
		llvm::SmallVector<Member, 3> members;
		llvm::SmallVector<Equation, 3> equations;
	};
}	 // namespace modelica
