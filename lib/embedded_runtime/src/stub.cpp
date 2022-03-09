
extern "C" int __cxa_guard_acquire(void *g){
    return 0;
}

extern "C" void __cxa_guard_release(void *g) noexcept
{
}

extern "C" void* __cxa_get_globals(){}

extern "C" void* __cxa_get_globals_fast(){}

void* pthread_once;

/*
void __cxa_rethrow(){
    return;
}

void __cxa_throw(){
    return;
}*/
