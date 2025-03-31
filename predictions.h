#pragma once

#include <stddef.h>
#include <stdint.h>

#define LABEL_PREFIX ("__label__")
#define LABEL_PREFIX_SIZE (sizeof(LABEL_PREFIX) - 1)

#define FREE_STRING(str)         \
    do                           \
    {                            \
        if (str.data != nullptr) \
            free(str.data);      \
        str.data = nullptr;      \
        str.size = 0;            \
    } while (0)

#ifdef __cplusplus
#define BEGIN_EXTERN_C() \
    extern "C"           \
    {
#else
#define BEGIN_EXTERN_C()
#endif

#ifdef __cplusplus
#define END_EXTERN_C() }
#else
#define END_EXTERN_C()
#endif

BEGIN_EXTERN_C()
typedef void *FastText_Handle_t;

typedef struct
{
    float *data;
    void *handle;
    size_t size;
} FastText_FloatVector_t;

typedef struct
{
    size_t size;
    char *data;
} FastText_String_t;

typedef struct
{
    float probability;
    FastText_String_t lang;
} FastText_PredictItem_t;

typedef struct
{
    enum
    {
        SUCCESS = 0,
        ERROR = 1,
    } status;

    union
    {
        FastText_Handle_t handle;
        char *error;
    };
} FastText_Result_t;

FastText_Result_t FastText_NewHandle(FastText_String_t path);
void FastText_DeleteHandle(const FastText_Handle_t handle);
size_t FastText_Predict(const FastText_Handle_t handle, FastText_String_t query, uint32_t k, float threshold,
                        FastText_PredictItem_t *const value);
FastText_FloatVector_t FastText_Wordvec(const FastText_Handle_t handle, FastText_String_t word);
FastText_FloatVector_t FastText_Sentencevec(const FastText_Handle_t handle, FastText_String_t sentence);

void FastText_FreeFloatVector(FastText_FloatVector_t vector);

END_EXTERN_C()
