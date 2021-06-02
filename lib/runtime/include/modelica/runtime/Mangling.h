#pragma once

#define NUM_ARGS_H1(dummy, x6, x5, x4, x3, x2, x1, x0, ...) x0
#define NUM_ARGS(...) NUM_ARGS_H1(dummy, ##__VA_ARGS__, 6, 5, 4, 3, 2, 1, 0)

#define APPLY0(t, n, dummy)
#define APPLY1(t, n, a) t(n, a)
#define APPLY2(t, n, a, b) APPLY1(t, 1, a), t(n, b)
#define APPLY3(t, n, a, b, c) APPLY2(t, 2, a, b), t(n, c)
#define APPLY4(t, n, a, b, c, d) APPLY3(t, 3, a, b, c), t(n, d)
#define APPLY5(t, n, a, b, c, d, e) APPLY4(t, 4, a, b, c, d), t(n, e)
#define APPLY6(t, n, a, b, c, d, e, f) APPLY5(t, 5, a, b, c, d, e), t(n, f)

#define APPLY_ALL_H3(t, n, ...) APPLY##n(t, n, __VA_ARGS__)
#define APPLY_ALL_H2(t, n, ...) APPLY_ALL_H3(t, n, __VA_ARGS__)
#define APPLY_ALL(t, ...) APPLY_ALL_H2(t, NUM_ARGS(__VA_ARGS__), __VA_ARGS__)

#define CONCAT0(dummy)
#define CONCAT1(a) a
#define CONCAT2(a, b) a ##b
#define CONCAT3(a, b, c) a ##b ##c
#define CONCAT4(a, b, c, d) a ##b ##c ##d
#define CONCAT5(a, b, c, d, e) a ##b ##c ##d ##e
#define CONCAT6(a, b, c, d, e, f) a ##b ##c ##d ##e ##f

#define CONCAT_ALL_H3(n, ...) CONCAT##n(__VA_ARGS__)
#define CONCAT_ALL_H2(n, ...) CONCAT_ALL_H3(n, __VA_ARGS__)
#define CONCAT_ALL(...) CONCAT_ALL_H2(NUM_ARGS(__VA_ARGS__), __VA_ARGS__)

#define ARRAY(type) ARRAY_ ##type

#define bool_CPP bool
#define int_CPP int
#define long_CPP long
#define float_CPP float
#define double_CPP double

#define ARRAY_bool_CPP UnsizedArrayDescriptor<bool_CPP>
#define ARRAY_int_CPP UnsizedArrayDescriptor<int_CPP>
#define ARRAY_long_CPP UnsizedArrayDescriptor<long_CPP>
#define ARRAY_float_CPP UnsizedArrayDescriptor<float_CPP>
#define ARRAY_double_CPP UnsizedArrayDescriptor<double_CPP>

#define TYPE_CPP(n, type) type ##_CPP
#define TYPES_CPP(...) APPLY_ALL(TYPE_CPP, __VA_ARGS__)

#define bool_MANGLED _i1
#define int_MANGLED _i32
#define long_MANGLED _i64
#define float_MANGLED _f32
#define double_MANGLED _f64

#define ARRAY_bool_MANGLED _ai1
#define ARRAY_int_MANGLED _ai32
#define ARRAY_long_MANGLED _ai64
#define ARRAY_float_MANGLED _af32
#define ARRAY_double_MANGLED _af64

#define TYPE_MANGLED(n, type) type ##_MANGLED
#define TYPES_MANGLED(...) APPLY_ALL(TYPE_MANGLED, __VA_ARGS__)

#define NAME_MANGLED(name, ...) CONCAT_ALL(_M, name, TYPES_MANGLED(__VA_ARGS__))

#define ARG_NAME(n, type) arg ##n
#define ARGS_NAMES(...) APPLY_ALL(ARG_NAME, __VA_ARGS__)

#define ARG_DECLARATION(n, type) type ARG_NAME(n, type)
#define ARGS_DECLARATIONS(...) APPLY_ALL(ARG_DECLARATION, __VA_ARGS__)

#define MLIR_PREFIX _mlir_ciface_

#define RUNTIME_FUNC_DECL(name, returnType, ...) \
	extern "C" returnType NAME_MANGLED(name, __VA_ARGS__) (ARGS_DECLARATIONS(TYPES_CPP(__VA_ARGS__))); \
  extern "C" returnType CONCAT_ALL(MLIR_PREFIX, NAME_MANGLED(name, __VA_ARGS__)) (ARGS_DECLARATIONS(TYPES_CPP(__VA_ARGS__)));

#define RUNTIME_FUNC_DEF(name, returnType, ...) \
	returnType NAME_MANGLED(name, __VA_ARGS__) (ARGS_DECLARATIONS(TYPES_CPP(__VA_ARGS__))) \
	{ \
		return name(ARGS_NAMES(__VA_ARGS__)); \
	} \
	returnType CONCAT_ALL(MLIR_PREFIX, NAME_MANGLED(name, __VA_ARGS__)) (ARGS_DECLARATIONS(TYPES_CPP(__VA_ARGS__))) \
	{ \
		return name(ARGS_NAMES(__VA_ARGS__)); \
	}
