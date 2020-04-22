#pragma once
#include "ModExp.hpp"

namespace modelica
{
	class ModEqTemplate
	{
		public:
		ModEqTemplate(ModExp left, ModExp right)
				: leftHand(std::move(left)), rightHand(std::move(right))
		{
		}

		[[nodiscard]] const ModExp& getLeft() const { return leftHand; }
		[[nodiscard]] const ModExp& getRight() const { return rightHand; }
		[[nodiscard]] ModExp& getLeft() { return leftHand; }
		[[nodiscard]] ModExp& getRight() { return rightHand; }

		private:
		ModExp leftHand;
		ModExp rightHand;
	};
}	 // namespace modelica
