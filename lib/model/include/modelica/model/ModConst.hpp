#pragma once

#include <utility>
#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/ModType.hpp"

namespace modelica
{
	/**
	 * A ModConst is a way to indicate values known at compile times
	 * They are the basic blocks of expressions.
	 *
	 * There are 3 template specialization that modelica knows how to
	 * lower, and they are one for each foudamental type, int, bool and float.
	 *
	 * Notice that a Const can be an array, but in that case the const is not able
	 * to have multiple dimensions, it's always just a vector and it's up to
	 * the user to determin how the array best fits the type he need.
	 */
	class ModConst
	{
		public:
		template<typename T>
		using Content = llvm::SmallVector<T, 3>;
		/**
		 * Builds a single value constant by providing that value.
		 */
		template<typename T>
		explicit ModConst(T val): content(Content<T>({ val }))
		{
		}

		template<typename T>
		ModConst(Content<T> args): content(std::move(args))
		{
		}

		ModConst(double d): content(Content<float>({ static_cast<float>(d) })) {}

		template<typename First, typename... T>
		explicit ModConst(First f, T&&... args)
				: content(Content<First>({ f, std::forward<T>(args)... }))
		{
		}

		/**
		 * \require index < size()
		 *
		 * \return the element at the indexth position.
		 */
		template<typename T>
		[[nodiscard]] T get(size_t index) const
		{
			const auto& cont = std::get<Content<T>>(content);
			assert(index < size());	 // NOLINT
			return cont[index];
		}

		template<typename T>
		[[nodiscard]] const Content<T>& getContent() const
		{
			assert(isA<T>());
			return std::get<Content<T>>(content);
		}

		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<Content<T>>(content);
		}

		[[nodiscard]] ModType getModTypeOfLiteral() const
		{
			return ModType(getBuiltinType(), size());
		}

		[[nodiscard]] BultinModTypes getBuiltinType() const
		{
			if (isA<int>())
				return BultinModTypes::INT;
			if (isA<bool>())
				return BultinModTypes::BOOL;
			if (isA<float>())
				return BultinModTypes::FLOAT;
			assert(false && "unreachable");
			return BultinModTypes::INT;
		}

		[[nodiscard]] size_t size() const
		{
			if (isA<int>())
				return getContent<int>().size();
			if (isA<bool>())
				return getContent<bool>().size();
			if (isA<float>())
				return getContent<float>().size();
			assert(false && "unreachable");
			return 0;
		}

		/**
		 * \return true iff every element is equal and have the same size
		 */
		bool operator==(const ModConst& other) const
		{
			return other.content == content;
		}
		/**
		 * \return Negation of operator ==
		 */
		bool operator!=(const ModConst& other) const { return !(*this == other); }

		void dump(llvm::raw_ostream& OS = llvm::outs()) const
		{
			OS << '{';
			for (size_t a = 0; a < size(); a++)
			{
				if (isA<int>())
					OS << get<int>(a);
				if (isA<float>())
					OS << get<float>(a);
				if (isA<bool>())
					OS << static_cast<int>(get<bool>(a));
				if (a != size() - 1)
					OS << ", ";
			}
			OS << '}';
		}

		private:
		/**
		 * The decision of selecting 3 in this small vector is totally arbitrary,
		 * i just assumed that 3d vectors are more likelly than everything else.
		 * May need profiling.
		 */
		std::variant<Content<int>, Content<float>, Content<bool>> content;
	};

	/**
	 * This template is used to check if
	 * a class is a instance of a template.
	 *
	 * This is usefull to determin the kind of
	 * constant you are receiving.
	 */
	template<class, template<class> class>
	struct is_instance: public std::false_type	// NOLINT
	{
	};

	template<class T, template<class> class U>
	struct is_instance<U<T>, U>: public std::true_type
	{
	};

}	 // namespace modelica
