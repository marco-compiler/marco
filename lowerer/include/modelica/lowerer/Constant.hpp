#pragma once

#include <variant>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace modelica
{
	template<typename C>
	class Constant
	{
		public:
		Constant(C val): content({ val }) {}

		template<typename... T2>
		Constant(C val, T2... args): content({ val, args... })
		{
		}

		[[nodiscard]] C get(size_t index) const
		{
			assert(index < content.size());	// NOLINT
			return content[index];
		}

		[[nodiscard]] size_t size() const { return content.size(); }

		bool operator==(const Constant& other) const
		{
			return other.content == content;
		}
		bool operator!=(const Constant& other) const { return !(*this == other); }

		private:
		llvm::SmallVector<C, 3> content;
	};

	template<typename T>
	void dumpConstant(const T& constant, llvm::raw_ostream& OS = llvm::outs())
	{
		for (size_t a = 0; a < constant.size(); a++)
		{
			OS << constant.get(a);
			if (a != constant.size() - 1)
				OS << ", ";
		}
	}

	template<class, template<class> class>
	struct is_instance: public std::false_type	// NOLINT
	{
	};

	template<class T, template<class> class U>
	struct is_instance<U<T>, U>: public std::true_type
	{
	};
	using IntConstant = Constant<int>;
	using FloatConstant = Constant<float>;
	using BoolConstant = Constant<bool>;
}	// namespace modelica
