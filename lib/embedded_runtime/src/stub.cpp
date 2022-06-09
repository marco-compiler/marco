/**
 *This file contains stub functions in order to allow the miosix compiler to compile without problems.
 * 
 */
extern "C" int __cxa_guard_acquire(void *g){
    return 0;
}

extern "C" void __cxa_guard_release(void *g) noexcept
{
}

extern "C" void* __cxa_get_globals(){}

extern "C" void* __cxa_get_globals_fast(){}

extern "C" void* __cxa_type_match(){}

extern "C" void* __cxa_call_unexpected(){}

extern "C" void* __cxa_begin_cleanup(){}


void* pthread_once;
