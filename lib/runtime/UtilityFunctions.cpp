#include "marco/runtime/UtilityFunctions.h"
#ifndef WINDOWS_NOSTDLIB
#include <cstring>
#include <iostream>
#else
//#include "marco/runtime/Printing.h"
#endif
#include <cassert>

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
inline void clone_void(UnsizedArrayDescriptor<T> destination, UnsizedArrayDescriptor<U> source)
{
	assert(source.getNumElements() == destination.getNumElements());

  auto sourceIt = source.begin();
  auto destinationIt = destination.begin();

  for (size_t i = 0, e = source.getNumElements(); i < e; ++i) {
    *destinationIt = *sourceIt;

    ++sourceIt;
    ++destinationIt;
  }
}

// Optimization for arrays with the same type
template<typename T>
inline void clone_void(UnsizedArrayDescriptor<T> destination, UnsizedArrayDescriptor<T> source)
{
	auto destinationSize = destination.getNumElements();
	assert(source.getNumElements() == destinationSize);
	memcpy(destination.getData(), source.getData(), destinationSize * sizeof(T));
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
  #ifndef WINDOWS_NOSTDLIB
	std::cout << std::scientific << value << std::endl;
  #else
  printString("Unknown type\n");
  #endif
}

template<>
inline void print_void<bool>(bool value)
{
  #ifndef WINDOWS_NOSTDLIB
	std::cout << std::boolalpha << value << std::endl;
  #else
  printBool(value);
  #endif
}

#ifdef WINDOWS_NOSTDLIB
template<>
inline void print_void<int32_t>(int32_t value)
{
  printInt(value);
}

template<>
inline void print_void<int64_t>(int64_t value)
{
  printInt(value);
}

template<>
inline void print_void<float>(float value)
{
  printFloat(value);
}

template<>
inline void print_void<double>(double value)
{
  printDouble(value);
}
#endif

RUNTIME_FUNC_DEF(print, void, bool)
RUNTIME_FUNC_DEF(print, void, int32_t)
RUNTIME_FUNC_DEF(print, void, int64_t)
RUNTIME_FUNC_DEF(print, void, float)
RUNTIME_FUNC_DEF(print, void, double)

template<typename T>
inline void print_void(UnsizedArrayDescriptor<T> array)
{
  #ifndef WINDOWS_NOSTDLIB
	std::cout << std::scientific << array << std::endl;
  #else
  printUnsized(array);
  #endif
}

template<>
inline void print_void<bool>(UnsizedArrayDescriptor<bool> array)
{
  #ifndef WINDOWS_NOSTDLIB
  std::cout << std::boolalpha << array << std::endl;
  #else
  printUnsized(array);
  #endif
}

RUNTIME_FUNC_DEF(print, void, ARRAY(bool))
RUNTIME_FUNC_DEF(print, void, ARRAY(int32_t))
RUNTIME_FUNC_DEF(print, void, ARRAY(int64_t))
RUNTIME_FUNC_DEF(print, void, ARRAY(float))
RUNTIME_FUNC_DEF(print, void, ARRAY(double))
