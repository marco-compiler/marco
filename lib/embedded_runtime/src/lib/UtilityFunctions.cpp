#include "../../include/marco/lib/UtilityFunctions.h"
#include "../../include/marco/lib/StdFunctions.h"
#include "../../include/marco/lib/Print.h"


//===----------------------------------------------------------------------===//
// clone
//===----------------------------------------------------------------------===//

/// Clone an array into another one.
///
/// @tparam T 					destination array type
/// @tparam U 					source array type
/// @param destination  destination array
/// @param values 			source values
template<typename T, typename U>
inline void clone_void(UnsizedArrayDescriptor<T>* destination, UnsizedArrayDescriptor<U>* source)
{
	stde::assertt(source->getNumElements() == destination->getNumElements());

  auto sourceIt = source->begin();
  auto destinationIt = destination->begin();

  for (size_t i = 0, e = source->getNumElements(); i < e; ++i) {
    *destinationIt = *sourceIt;

    ++sourceIt;
    ++destinationIt;
  }
}

// Optimization for arrays with the same type
template<typename T>
inline void clone_void(UnsizedArrayDescriptor<T>* destination, UnsizedArrayDescriptor<T>* source)
{
	auto destinationSize = destination->getNumElements();
	stde::assertt(source->getNumElements() == destinationSize);
	stde::memcpy(destination->getData(), source->getData(), destinationSize * sizeof(T));
}

RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int32_t), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(int64_t), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(int32_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(int64_t))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DEF(clone, void, ARRAY(double), ARRAY(double))

//===----------------------------------------------------------------------===//
// print
//===----------------------------------------------------------------------===//

template<typename T>
inline void print_void(T value)
{
  print_char("Unknown type\n");
}

template<>
inline void print_void<bool>(bool value)
{
  print_integer(value);
}

template<>
inline void print_void<int32_t>(int32_t value)
{
  print_integer(value);
  print_char("\n\r");
}

template<>
inline void print_void<int64_t>(int64_t value)
{
  print_integer(value);
  print_char("\n\r");
}

template<>
inline void print_void<float>(float value)
{
  print_float(value);
  print_char("\n\r");
}

template<>
inline void print_void<double>(double value)
{
    print_float(value);
  print_char("\n\r");
}


RUNTIME_FUNC_DEF(print, void, bool)
RUNTIME_FUNC_DEF(print, void, int32_t)
RUNTIME_FUNC_DEF(print, void, int64_t)
RUNTIME_FUNC_DEF(print, void, float)
RUNTIME_FUNC_DEF(print, void, double)

template<typename T>
inline void print_void(UnsizedArrayDescriptor<T>* array)
{

  for(auto a : *array){
    print_serial(a);
  }

}

template<>
inline void print_void<bool>(UnsizedArrayDescriptor<bool>* array)
{
  for(bool a : *array){
    print_integer(a);
  }
}

RUNTIME_FUNC_DEF(print, void, ARRAY(bool))
RUNTIME_FUNC_DEF(print, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(print, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(print, void, ARRAY(float))
RUNTIME_FUNC_DEF(print, void, ARRAY(double))