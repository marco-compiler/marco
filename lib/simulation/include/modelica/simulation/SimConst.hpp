#pragma once

#include <variant>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace modelica
{
	/**
	 * A SimConst is a way to indicate values known at compile times
	 * They are the basic blocks of expressions.
	 *
	 * There are 3 template specialization that modelica knows how to
	 * lower, and they are one for each foudamental type, int, bool and float.
	 *
	 * Notice that a Const can be an array, but in that case the const is not able
	 * to have multiple dimensions, it's always just a vector and it's up to
	 * the user to determin how the array best fits the type he need.
	 */
	template<typename C>
	class SimConst
	{
		public:
		/**
		 * Builds a single value constant by providing that value.
		 */
		SimConst(C val): content({ val }) {}

		/**
		 * Builds a constant by specifiying every value.
		 */
		template<typename... T2>
		SimConst(C val, T2... args): content({ val, args... })
		{
		}

		/**
		 * \require index < size()
		 *
		 * \return the element at the indexth position.
		 */
		[[nodiscard]] C get(size_t index) const
		{
			assert(index < content.size());	 // NOLINT
			return content[index];
		}

		[[nodiscard]] size_t size() const { return content.size(); }

		/**
		 * \return true iff every element is equal and have the same size
		 */
		bool operator==(const SimConst& other) const
		{
			return other.content == content;
		}
		/**
		 * \return Negation of operator ==
		 */
		bool operator!=(const SimConst& other) const { return !(*this == other); }

		private:
		/**
		 * The decision of selecting 3 in this small vector is totally arbitrary,
		 * i just assumed that 3d vectors are more likelly than everything else.
		 * May need profiling.
		 */
		llvm::SmallVector<C, 3> content;
	};

	/**
	 * Dumps a constant values onto the provided outputstream,
	 * llvm::outs() by default
	 */
	template<typename T>
	void dumpConstant(const T& constant, llvm::raw_ostream& OS = llvm::outs())
	{
		OS << '{';
		for (size_t a = 0; a < constant.size(); a++)
		{
			OS << constant.get(a);
			if (a != constant.size() - 1)
				OS << ", ";
		}
		OS << '}';
	}

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

	using IntSimConst = SimConst<int>;
	using FloatSimConst = SimConst<float>;
	using BoolSimConst = SimConst<bool>;
}	 // namespace modelica
