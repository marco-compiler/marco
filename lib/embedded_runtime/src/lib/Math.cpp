#include "../../include/marco/lib/Math.h"
#include "../../include/marco/lib/StdFunctions.h"
//#include <cassert>
//#include <cmath>

//===----------------------------------------------------------------------===//
// pow
//===----------------------------------------------------------------------===//
// TODO tests

template<typename Result, typename Base, typename Exp>
inline Result pow(Base base, Exp exp)
{
  if (base == 0) {
    stde::assertt(exp > 0);
    return base;
  }

  if (exp == 0) {
    return 1;
  }

  return stde::pow(base, exp);
}

template<typename Exp>
inline bool pow_boolBase(bool base, Exp exp)
{
  if (base) {
    return true;
  }

  stde::assertt(exp > 0);
  return false;
}

template<>
inline bool pow_boolBase<bool>(bool base, bool exp)
{
  stde::assertt(base || exp);
  return base;
}

template<typename Result, typename Base>
inline Result pow_boolExp(Base base, bool exp)
{
  if (exp) {
    return base;
  }

  stde::assertt(base != 0);
  return 1;
}

template<>
inline bool pow_boolExp(bool base, bool exp)
{
  return pow_boolBase(base, exp);
}

template<typename Base, typename Exp>
inline bool pow_i1(Base base, Exp exp)
{
  stde::assertt(base != 0 || exp != 0);

  if (base == 0) {
    return false;
  }

  return true;
}

inline bool pow_i1(bool base, bool exp)
{
  return pow_boolBase(base, exp);
}

inline bool pow_i1(bool base, int32_t exp)
{
  return pow_boolBase(base, exp);
}

inline bool pow_i1(bool base, int64_t exp)
{
  return pow_boolBase(base, exp);
}

inline bool pow_i1(bool base, float exp)
{
  return pow_boolBase(base, exp);
}

inline bool pow_i1(bool base, double exp)
{
  return pow_boolBase(base, exp);
}

inline bool pow_i1(int32_t base, bool exp)
{
  return pow_boolExp<int32_t>(base, exp) > 0;
}

inline bool pow_i1(int64_t base, bool exp)
{
  return pow_boolExp<int64_t>(base, exp) > 0;
}

inline bool pow_i1(float base, bool exp)
{
  return pow_boolExp<float>(base, exp) > 0;
}

inline bool pow_i1(double base, bool exp)
{
  return pow_boolExp<double>(base, exp) > 0;
}

RUNTIME_FUNC_DEF(pow, bool, bool, bool)
RUNTIME_FUNC_DEF(pow, bool, bool, int32_t)
RUNTIME_FUNC_DEF(pow, bool, bool, int64_t)
RUNTIME_FUNC_DEF(pow, bool, bool, float)
RUNTIME_FUNC_DEF(pow, bool, bool, double)

RUNTIME_FUNC_DEF(pow, bool, int32_t, bool)
RUNTIME_FUNC_DEF(pow, bool, int32_t, int32_t)
RUNTIME_FUNC_DEF(pow, bool, int32_t, float)

RUNTIME_FUNC_DEF(pow, bool, int64_t, bool)
RUNTIME_FUNC_DEF(pow, bool, int64_t, int64_t)
RUNTIME_FUNC_DEF(pow, bool, int64_t, double)

RUNTIME_FUNC_DEF(pow, bool, float, bool)
RUNTIME_FUNC_DEF(pow, bool, float, int32_t)
RUNTIME_FUNC_DEF(pow, bool, float, float)

RUNTIME_FUNC_DEF(pow, bool, double, bool)
RUNTIME_FUNC_DEF(pow, bool, double, int64_t)
RUNTIME_FUNC_DEF(pow, bool, double, double)

template<typename Base, typename Exp>
inline int32_t pow_i32(Base base, Exp exp)
{
  return stde::pow(base, exp);
}

template<>
inline int32_t pow_i32<bool, bool>(bool base, bool exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline int32_t pow_i32<bool, int32_t>(bool base, int32_t exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline int32_t pow_i32<bool, float>(bool base, float exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline int32_t pow_i32<int32_t, bool>(int32_t base, bool exp)
{
  return pow_boolExp<int32_t>(base, exp);
}

template<>
inline int32_t pow_i32<float, bool>(float base, bool exp)
{
  return pow_boolExp<float>(base, exp);
}

RUNTIME_FUNC_DEF(pow, int32_t, bool, bool)
RUNTIME_FUNC_DEF(pow, int32_t, bool, int32_t)
RUNTIME_FUNC_DEF(pow, int32_t, bool, float)

RUNTIME_FUNC_DEF(pow, int32_t, int32_t, bool)
RUNTIME_FUNC_DEF(pow, int32_t, int32_t, int32_t)
RUNTIME_FUNC_DEF(pow, int32_t, int32_t, float)

RUNTIME_FUNC_DEF(pow, int32_t, float, bool)
RUNTIME_FUNC_DEF(pow, int32_t, float, int32_t)
RUNTIME_FUNC_DEF(pow, int32_t, float, float)

template<typename Base, typename Exp>
inline int64_t pow_i64(Base base, Exp exp)
{
  return stde::pow(base, exp);
}

template<>
inline int64_t pow_i64<bool, bool>(bool base, bool exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline int64_t pow_i64<bool, int64_t>(bool base, int64_t exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline int64_t pow_i64<bool, double>(bool base, double exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline int64_t pow_i64<int64_t, bool>(int64_t base, bool exp)
{
  return pow_boolExp<int64_t>(base, exp);
}

template<>
inline int64_t pow_i64<double, bool>(double base, bool exp)
{
  return pow_boolExp<double>(base, exp);
}

RUNTIME_FUNC_DEF(pow, int64_t, bool, bool)
RUNTIME_FUNC_DEF(pow, int64_t, bool, int64_t)
RUNTIME_FUNC_DEF(pow, int64_t, bool, double)

RUNTIME_FUNC_DEF(pow, int64_t, int64_t, bool)
RUNTIME_FUNC_DEF(pow, int64_t, int64_t, int64_t)
RUNTIME_FUNC_DEF(pow, int64_t, int64_t, double)

RUNTIME_FUNC_DEF(pow, int64_t, double, bool)
RUNTIME_FUNC_DEF(pow, int64_t, double, int64_t)
RUNTIME_FUNC_DEF(pow, int64_t, double, double)

template<typename Base, typename Exp>
inline float pow_f32(Base base, Exp exp)
{
  return stde::pow(base, exp);
}

template<>
inline float pow_f32<bool, bool>(bool base, bool exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline float pow_f32<bool, int32_t>(bool base, int32_t exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline float pow_f32<bool, float>(bool base, float exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline float pow_f32<int32_t, bool>(int32_t base, bool exp)
{
  return pow_boolExp<int32_t>(base, exp);
}

template<>
inline float pow_f32<float, bool>(float base, bool exp)
{
  return pow_boolExp<float>(base, exp);
}

RUNTIME_FUNC_DEF(pow, float, bool, bool)
RUNTIME_FUNC_DEF(pow, float, bool, int32_t)
RUNTIME_FUNC_DEF(pow, float, bool, float)

RUNTIME_FUNC_DEF(pow, float, int32_t, bool)
RUNTIME_FUNC_DEF(pow, float, int32_t, int32_t)
RUNTIME_FUNC_DEF(pow, float, int32_t, float)

RUNTIME_FUNC_DEF(pow, float, float, bool)
RUNTIME_FUNC_DEF(pow, float, float, int32_t)
RUNTIME_FUNC_DEF(pow, float, float, float)

template<typename Base, typename Exp>
inline double pow_f64(Base base, Exp exp)
{
  return stde::pow(base, exp);
}

template<>
inline double pow_f64<bool, bool>(bool base, bool exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline double pow_f64<bool, int64_t>(bool base, int64_t exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline double pow_f64<bool, double>(bool base, double exp)
{
  return pow_boolBase(base, exp) ? 1 : 0;
}

template<>
inline double pow_f64<int64_t, bool>(int64_t base, bool exp)
{
  return pow_boolExp<int64_t>(base, exp);
}

template<>
inline double pow_f64<double, bool>(double base, bool exp)
{
  return pow_boolExp<double>(base, exp);
}

RUNTIME_FUNC_DEF(pow, double, bool, bool)
RUNTIME_FUNC_DEF(pow, double, bool, int64_t)
RUNTIME_FUNC_DEF(pow, double, bool, double)

RUNTIME_FUNC_DEF(pow, double, int32_t, bool)
RUNTIME_FUNC_DEF(pow, double, int32_t, int64_t)
RUNTIME_FUNC_DEF(pow, double, int32_t, double)

RUNTIME_FUNC_DEF(pow, double, int64_t, bool)
RUNTIME_FUNC_DEF(pow, double, int64_t, int64_t)
RUNTIME_FUNC_DEF(pow, double, int64_t, double)

RUNTIME_FUNC_DEF(pow, double, double, bool)
RUNTIME_FUNC_DEF(pow, double, double, int64_t)
RUNTIME_FUNC_DEF(pow, double, double, double)
