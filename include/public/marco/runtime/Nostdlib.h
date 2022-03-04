extern "C" __declspec(dllexport) void* memmove(void* dstpp, const void* srcpp, size_t len);
extern "C" __declspec(dllexport) void* memcpy(void* dstpp, const void* srcpp, size_t len);
extern "C" __declspec(dllexport) void* memset(void* s, int c,  size_t len);
extern "C" __declspec(dllexport) void runtimeMemset(char *p, char c, int l);