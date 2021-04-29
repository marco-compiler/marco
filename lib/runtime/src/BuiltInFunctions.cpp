#include <modelica/runtime/Runtime.h>
#include <iostream>

template<typename T>
inline void fill(UnsizedArrayDescriptor<T> descriptor, T value)
{
	for (auto& element : descriptor)
		element = value;
}

void _Mfill_i1(UnsizedArrayDescriptor<bool> descriptor, bool value)
{
	fill(descriptor, value);
}

void _Mfill_i32(UnsizedArrayDescriptor<int> descriptor, int value)
{
	fill(descriptor, value);
}

void _Mfill_i64(UnsizedArrayDescriptor<long> descriptor, long value)
{
	fill(descriptor, value);
}

void _Mfill_f32(UnsizedArrayDescriptor<float> descriptor, float value)
{
	fill(descriptor, value);
}

void _Mfill_f64(UnsizedArrayDescriptor<double> descriptor, double value)
{
	fill(descriptor, value);
}
