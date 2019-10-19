#pragma once

#include "modelica/model/ModExp.hpp"

namespace modelica
{
	class ModEquation
	{
		public:
		ModEquation(ModExp left, ModExp right)
				: leftHand(std::make_unique<ModExp>(std::move(left))),
					rightHand(std::make_unique<ModExp>(std::move(right)))
		{
			assert(leftHand != nullptr);	 // NOLINT
			assert(rightHand != nullptr);	 // NOLINT
		}

		[[nodiscard]] const ModExp& getLeft() const { return *leftHand; }
		[[nodiscard]] const ModExp& getRight() const { return *rightHand; }
		[[nodiscard]] ModExp& getLeft() { return *leftHand; }
		[[nodiscard]] ModExp& getRight() { return *rightHand; }

		private:
		std::unique_ptr<ModExp> leftHand;
		std::unique_ptr<ModExp> rightHand;
	};
}	 // namespace modelica
